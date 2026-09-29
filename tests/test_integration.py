"""Real GPU environments, local DINO rendering, and complete PPO round trips."""

import csv
import gc
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import optax
import pytest
import torch
from ml_collections import ConfigDict
from mujoco_playground import registry
from mujoco_playground._src import wrapper

import benchmark  # noqa: F401
from benchmark.common.dino import dino_config
from benchmark.common.envs import DEFAULT_TASK, GOAL_BODIES, TASKS, NonVisionWrapper, env_config, make_env
from benchmark.common.mapping import MapObservationWrapper, components, quat_matrix, visual_mesh
from benchmark.common.policy import load_policy
from benchmark.eval import evaluate

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
    gc.collect()


def map_settings():
    # Reduced view count tests the real pipeline without a full production scan.
    return {"background": False, "voxel_size": .015,
            "views": 4, "extra_views": 0, "cache": str(Path('.cache/maps').resolve()), "dino": dino_config()}


@pytest.mark.parametrize("task", TASKS)
def test_task_native_and_map_physics(task, monkeypatch):
    config = env_config(task, "state")
    native = registry.load(task, config=ConfigDict(config['environment']))
    if hasattr(native, 'defer_rendering'):
        native = NonVisionWrapper(native)
    hidden = make_env(config)
    model_arrays = {name: getattr(hidden.mj_model, name).copy() for name in
                    ('qpos0', 'geom_pos', 'geom_quat', 'geom_rgba', 'geom_group', 'light_active')}
    torch_rng = torch.get_rng_state().clone()
    mapper = MapObservationWrapper(hidden, task, map_settings())
    assert torch.equal(torch_rng, torch.get_rng_state())
    for name, value in model_arrays.items():
        np.testing.assert_array_equal(getattr(hidden.mj_model, name), value)
    assert len(mapper.feature_ids) > 0
    assert np.isfinite(mapper.local).all()
    assert not any(name.startswith("world/") for name in mapper.component_names) or task.startswith("Aloha")
    assert not set(GOAL_BODIES).intersection(mapper.component_names)
    # A cache hit must not instantiate DINO or write a new template.
    with monkeypatch.context() as patch:
        patch.setattr('benchmark.common.mapping.build_template', lambda *a, **kw: pytest.fail("Cache miss on identical model"))
        cached = MapObservationWrapper(hidden, task, map_settings())
        assert cached.bank.paths == mapper.bank.paths
        np.testing.assert_array_equal(cached.local, mapper.local)
    repeat = config["environment"]["action_repeat"]
    # A short external timeout exercises the native cached autoreset/history path.
    raw = wrapper.wrap_for_brax_training(native, episode_length=2 * repeat, action_repeat=repeat)
    mapped = wrapper.wrap_for_brax_training(mapper, episode_length=2 * repeat, action_repeat=repeat)
    keys = jax.random.split(jax.random.PRNGKey(13), 2)
    a = jax.jit(raw.reset)(keys)
    b = jax.jit(mapped.reset)(keys)
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.info['rng'], b.info['rng'])
    raw_step, map_step = jax.jit(raw.step), jax.jit(mapped.step)
    action = jnp.full((2, native.action_size), .05)
    for _ in range(5):
        a, b = raw_step(a, action), map_step(b, action)
        # Identical native Warp runs also vary on RTX 4090. Measured maxima:
        # Leap: 1.2e-5 qpos, 4.6e-4 qvel, 9.7e-4 reward; Push: 4.5e-5 qvel.
        tolerances = {'LeapCubeRotateZAxis': (2e-5, 1e-3, 2e-3),
                      'PandaRobotiqPushCube': (1e-6, 1e-4, 1e-5)}.get(task, (1e-6, 1e-6, 1e-5))
        for (x, y), atol in zip(((a.data.qpos, b.data.qpos), (a.data.qvel, b.data.qvel), (a.reward, b.reward)), tolerances):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=atol)
        np.testing.assert_array_equal(a.done, b.done)
        np.testing.assert_array_equal(a.info['rng'], b.info['rng'])
        expected = jax.vmap(mapper.observation)(a.data)
        np.testing.assert_allclose(b.obs['xyz'], expected['xyz'], rtol=1e-5, atol=2e-6)
        np.testing.assert_array_equal(b.obs['feature_ids'], expected['feature_ids'])
        assert np.isfinite(b.obs['xyz']).all()
    print(f"PASS {task}: {native.action_size} actions, {len(mapper.feature_ids)} map points", flush=True)


@pytest.mark.parametrize("task", TASKS)
def test_task_visual_parts_and_goal_removal(task):
    native = registry.load(task)
    hidden = make_env(env_config(task))
    original, model = native.mj_model, hidden.mj_model
    # Only render fields change: even goal collision proxies retain their physics.
    for name in ('geom_contype', 'geom_conaffinity', 'body_mass', 'body_inertia',
                 'geom_pos', 'geom_quat', 'geom_size', 'body_pos', 'qpos0', 'mat_rgba'):
        np.testing.assert_array_equal(getattr(model, name), getattr(original, name))
    goal = np.array([model.body(int(body)).name in GOAL_BODIES for body in model.geom_bodyid])
    assert np.all(model.geom_group[goal] == 5)
    assert np.all(model.geom_rgba[goal, 3] == 0)
    assert np.all(model.geom_matid[goal] == -1)
    for name in ('geom_group', 'geom_rgba', 'geom_matid'):
        np.testing.assert_array_equal(getattr(model, name)[~goal], getattr(original, name)[~goal])
    groups = components(model, task, {'background': False})
    assert groups == components(original, task, {'background': False})
    assert not any(model.body(body).name in GOAL_BODIES for body, _, _ in groups)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    hinges = np.flatnonzero(np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]))
    moved = set()
    first = None
    for displacement in (0., .05):
        data.qpos[model.jnt_qposadr[hinges]] += displacement
        mujoco.mj_forward(model, data)
        if first is None:
            first = data.xpos.copy(), data.xmat.copy()
        else:
            moved = {body for body, _, _ in groups if
                     not np.allclose(data.xpos[body], first[0][body]) or
                     not np.allclose(data.xmat[body], first[1][body])}
        # Body-local map vertices must agree with independently transformed geom vertices.
        for body, geoms, _ in groups:
            for geom in geoms:
                local = np.asarray(visual_mesh(model, [geom]).vertices)[::17]
                geom_local = (local - model.geom_pos[geom]) @ quat_matrix(model.geom_quat[geom])
                expected = geom_local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                actual = local @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
                np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert len(moved) >= 2  # Arm/hand links really articulate separately.
    if task == 'PandaOpenCabinet':
        handle = model.body('handle').id
        assert handle in moved
        np.testing.assert_allclose(data.xpos[handle] - first[0][handle], [.05, 0, 0], atol=1e-7)


def test_native_rgb():
    native = make_env(env_config(DEFAULT_TASK, "rgb"), num_envs=2)
    env = wrapper.wrap_for_brax_training(native, episode_length=200, action_repeat=1)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    state = jax.jit(env.step)(state, jnp.zeros((2, 3)))
    assert set(state.obs) == {'pixels/view_0'}
    pixels = np.asarray(state.obs['pixels/view_0'])
    assert pixels.shape == (2, 64, 64, 3)
    assert np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1
    assert pixels.std() > 0
    goal = native.mj_model.body('mocap_target').id
    mocap = int(native.mj_model.body_mocapid[goal])

    @jax.jit
    def render_goal(position):
        data = state.data.replace(mocap_pos=state.data.mocap_pos.at[:, mocap].set(position))
        data = jax.vmap(lambda d: mjx.kinematics(native.mjx_model, d))(data)
        return native.render_state(state.replace(data=data)).obs['pixels/view_0']

    # Moving the hidden marker onto the visible cube must leave RGB unchanged.
    cube = native.mj_model.body('box').id
    near = render_goal(state.data.xpos[:, cube])
    far = render_goal(jnp.full((2, 3), 100.))
    np.testing.assert_array_equal(near, far)


@pytest.mark.parametrize('mode', ['state', 'rgb', 'map'])
def test_training_checkpoint_and_evaluation(tmp_path, monkeypatch, mode):
    train_module = importlib.import_module('benchmark.rl.train')
    real_train = train_module.ppo.train
    real_gradient = train_module.ppo.gradients.loss_and_pgrad
    snapshots = []
    inference = []

    def record(step, make_policy, params):
        snapshots.append(jax.device_get(params[1]))
        inference[:] = [make_policy(params, deterministic=True)]

    def checked_train(**kwargs):
        return real_train(**kwargs, policy_params_fn=record)

    def measured_gradient(*args, **kwargs):
        compute = real_gradient(*args, **kwargs)

        def measured(*values):
            (loss, metrics), grads = compute(*values)
            return (loss, {**metrics, 'test_grad_norm': optax.tree.norm(grads)}), grads

        return measured

    monkeypatch.setattr(train_module.ppo, 'train', checked_train)
    monkeypatch.setattr(train_module.ppo.gradients, 'loss_and_pgrad', measured_gradient)
    args = train_module.parser().parse_args([
        '--obs-mode', mode, '--num-envs', '2', '--num-eval-envs', '1',
        '--total-timesteps', '8', '--unroll-length', '2', '--batch-size', '2',
        '--num-minibatches', '1', '--num-updates-per-batch', '1', '--num-evals', '2',
        '--map-views', '4', '--map-extra-views', '0',
        '--output', str(tmp_path / mode),
    ])
    run = train_module.train(args)
    with open(run / 'train.csv') as stream:
        metrics = list(csv.DictReader(stream))
    for name in ('training/total_loss', 'training/test_grad_norm'):
        values = [float(row['value']) for row in metrics if row['metric'] == name]
        assert values and np.isfinite(values).all()
    assert any(value > 0 for value in values)
    assert len(snapshots) >= 2
    before, after = jax.tree.leaves(snapshots[0]), jax.tree.leaves(snapshots[-1])
    assert all(np.isfinite(x).all() for x in after)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    policy = load_policy(run)
    env = make_env(policy.env_config, 1)
    env = wrapper.wrap_for_brax_training(env, policy.metadata['ppo']['episode_length'], policy.metadata['ppo']['action_repeat'])
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(7), 1))
    expected = jax.jit(inference[0])(state.obs, jax.random.PRNGKey(0))[0]
    np.testing.assert_allclose(policy.act(state.obs), expected, rtol=1e-6, atol=1e-6)
    summary, rows = evaluate(policy, episodes=3, num_envs=2)
    assert summary['episodes'] == len(rows) == 3
    assert all(np.isfinite(value) for row in rows for value in row.values())
    assert all(row['episode_len'] <= policy.metadata['ppo']['episode_length'] for row in rows)
