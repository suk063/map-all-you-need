"""Real GPU environments, local DINO rendering, and complete PPO round trips."""

import csv
import gc
import importlib
import json
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import pytest
import torch
from ml_collections import ConfigDict
from mujoco_playground import registry
from mujoco_playground._src import wrapper

import benchmark  # noqa: F401
from benchmark.common.dino import dino_config
from benchmark.common.envs import (
    DEFAULT_TASK,
    GOAL_BODIES,
    GOAL_TASKS,
    TASKS,
    NonVisionWrapper,
    env_config,
    make_env,
)
from benchmark.common.mapping import (
    MapObservationWrapper,
    components,
    quat_matrix,
    visual_mesh,
)
from benchmark.common.policy import latest_checkpoint, load_policy
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
    base = make_env(config)
    model_arrays = {name: getattr(base.mj_model, name).copy() for name in
                    ('qpos0', 'geom_pos', 'geom_quat', 'geom_rgba', 'geom_group', 'light_active')}
    torch_rng = torch.get_rng_state().clone()
    mapper = MapObservationWrapper(base, task, map_settings())
    assert torch.equal(torch_rng, torch.get_rng_state())
    for name, value in model_arrays.items():
        np.testing.assert_array_equal(getattr(base.mj_model, name), value)
    assert len(mapper.feature_ids) > 0
    assert np.isfinite(mapper.local).all()
    np.testing.assert_allclose(np.linalg.norm(mapper.local_normals, axis=-1), 1, atol=1e-5)
    assert not any(name.startswith("world/") for name in mapper.component_names) or task.startswith("Aloha")
    expected_goal = (set() if task in ('AlohaSinglePegInsertion', 'LeapCubeRotateZAxis', 'AeroCubeRotateZAxis')
                     else {'goal' if task == 'LeapCubeReorient' else 'mocap_target'})
    assert set(GOAL_BODIES).intersection(mapper.component_names) == expected_goal
    # A cache hit must not instantiate DINO or write a new template.
    with monkeypatch.context() as patch:
        patch.setattr('benchmark.common.mapping.build_template', lambda *a, **kw: pytest.fail("Cache miss on identical model"))
        cached = MapObservationWrapper(base, task, map_settings())
        assert cached.bank.paths == mapper.bank.paths
        np.testing.assert_array_equal(cached.local, mapper.local)
    repeat = config["environment"]["action_repeat"]
    # A short external timeout exercises the native cached autoreset/history path.
    raw = wrapper.wrap_for_brax_training(native, episode_length=2 * repeat, action_repeat=repeat)
    mapped = wrapper.wrap_for_brax_training(mapper, episode_length=2 * repeat, action_repeat=repeat)
    keys = jax.random.split(jax.random.PRNGKey(13), 2)
    a = jax.jit(raw.reset)(keys)
    b = jax.jit(mapped.reset)(keys)
    assert np.abs(b.obs['xyz']).max() <= 1 + 1e-6
    centers, scales = b.info['_map_center'], b.info['_map_scale']
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.info['rng'], b.info['rng'])
    for name in expected_goal:
        body = native.mj_model.body(name).id
        mocap = native.mj_model.body_mocapid[body]
        mask = np.asarray(mapper.body_ids) == body
        assert mask.any()
        local = np.asarray(mapper.local)[mask]
        for i in range(2):
            goal_xyz = local @ quat_matrix(np.asarray(a.data.mocap_quat[i, mocap])).T + np.asarray(a.data.mocap_pos[i, mocap])
            origin = np.asarray(a.data.xpos[i, mapper.frame_bodies]).mean(0)
            axes = np.asarray(a.data.xmat[i, mapper.frame_axes])
            goal_xyz = ((goal_xyz - origin) @ axes - centers[i]) / scales[i]
            np.testing.assert_allclose(np.asarray(b.obs['xyz']).reshape(2, -1, 3)[i, mask], goal_xyz, atol=1e-6)
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
        expected = jax.vmap(mapper.observation)(b.data, b.info)
        np.testing.assert_allclose(b.obs['xyz'], expected['xyz'], rtol=1e-5, atol=2e-6)
        np.testing.assert_array_equal(b.obs['feature_ids'], expected['feature_ids'])
        np.testing.assert_allclose(b.obs['normals'], expected['normals'], atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(b.obs['normals'].reshape(2, -1, 3), axis=-1), 1, atol=1e-5)
        np.testing.assert_array_equal(b.info['_map_center'], centers)
        np.testing.assert_array_equal(b.info['_map_scale'], scales)
        np.testing.assert_allclose(b.obs['geometry_epsilon'], 1e-6 / scales)
        assert np.isfinite(b.obs['xyz']).all()
    print(f"PASS {task}: {native.action_size} actions, {len(mapper.feature_ids)} map points", flush=True)


@pytest.mark.parametrize("task", TASKS)
def test_task_visual_parts_and_goal_visibility(task):
    native = registry.load(task)
    base = make_env(env_config(task))
    original, model = native.mj_model, base.mj_model
    # Hiding unused markers must not affect physics or shared object materials.
    for name in ('geom_contype', 'geom_conaffinity', 'body_mass', 'body_inertia',
                 'geom_pos', 'geom_quat', 'geom_size', 'body_pos', 'qpos0', 'mat_rgba'):
        np.testing.assert_array_equal(getattr(model, name), getattr(original, name))
    hidden = np.array([task not in GOAL_TASKS and model.body(int(b)).name in GOAL_BODIES for b in model.geom_bodyid])
    for name in ('geom_group', 'geom_rgba', 'geom_matid'):
        np.testing.assert_array_equal(getattr(model, name)[~hidden], getattr(original, name)[~hidden])
    assert np.all(model.geom_rgba[hidden, 3] == 0)
    assert np.all(model.geom_matid[hidden] == -1)
    assert np.all(model.geom_group[hidden] == 5)
    groups = components(model, task, {'background': False})
    assert groups == components(original, task, {'background': False})
    expected_goal = (set() if task in ('AlohaSinglePegInsertion', 'LeapCubeRotateZAxis', 'AeroCubeRotateZAxis')
                     else {'goal' if task == 'LeapCubeReorient' else 'mocap_target'})
    assert {model.body(body).name for body, _, _ in groups} & set(GOAL_BODIES) == expected_goal
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


@pytest.mark.parametrize('task', TASKS)
def test_native_rgb(task):
    config = env_config(task, 'rgb')
    native = make_env(config, num_envs=2)
    reference = make_env(env_config(task, 'state'))
    repeat = config['environment']['action_repeat']
    env = wrapper.wrap_for_brax_training(native, episode_length=repeat * 2, action_repeat=repeat)
    raw = wrapper.wrap_for_brax_training(reference, episode_length=repeat * 2, action_repeat=repeat)
    assert isinstance(env, wrapper.DeferredVisionWrapper)
    assert env.unwrapped is native
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    original = jax.jit(raw.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    rgb_step, state_step = jax.jit(env.step), jax.jit(raw.step)
    action = jnp.full((2, native.action_size), .05)
    for _ in range(5):
        state, original = rgb_step(state, action), state_step(original, action)
        tolerances = {'LeapCubeRotateZAxis': (2e-5, 1e-3, 2e-3),
                      'PandaRobotiqPushCube': (1e-6, 1e-4, 1e-5)}.get(task, (1e-6, 1e-6, 1e-5))
        for (x, y), atol in zip(((state.data.qpos, original.data.qpos), (state.data.qvel, original.data.qvel),
                                (state.reward, original.reward)), tolerances):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=atol)
        np.testing.assert_array_equal(state.done, original.done)
        np.testing.assert_array_equal(state.info['rng'], original.info['rng'])
        for x, y in zip(jax.tree.leaves(state.info['DeferredVisionWrapper_obs']), jax.tree.leaves(original.obs)):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=2e-3)
    assert set(state.obs) == {'pixels/view_0'}
    pixels = np.asarray(state.obs['pixels/view_0'])
    assert pixels.shape == (2, 64, 64, 3)
    assert pixels.dtype == np.float32
    assert np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1
    assert pixels.std() > 0
    assert native.render_metadata['use_textures']
    # Optional validation artifacts allow review of framing, not just shapes.
    if os.environ.get('RGB_VALIDATION_OUTPUT'):
        from PIL import Image
        directory = Path(os.environ['RGB_VALIDATION_OUTPUT'])
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.round(pixels[0] * 255).astype(np.uint8)).save(directory / f'{task}.png')
    # Rendering returns the exact native dynamics data, never a private camera model's data.
    rerendered = native.render_state(state)
    for x, y in zip(jax.tree.leaves(state.data), jax.tree.leaves(rerendered.data)):
        np.testing.assert_array_equal(x, y)
    if task == 'LeapCubeReorient':
        # Compare actual pixels, with identical physics/camera, against disabling
        # native textures. The orientation cube's face textures must survive.
        plain_config = {**config, 'rgb': {**config['rgb'], 'use_textures': False}}
        plain = make_env(plain_config, num_envs=2)
        untextured = np.asarray(jax.jit(plain.render_state)(state).obs['pixels/view_0'])
        assert np.mean(np.abs(pixels - untextured)) > 1e-4
        if os.environ.get('RGB_VALIDATION_OUTPUT'):
            Image.fromarray(np.round(untextured[0] * 255).astype(np.uint8)).save(directory / f'{task}-no-textures.png')
    if task != DEFAULT_TASK:
        return
    goal = native.mj_model.body('mocap_target').id
    mocap = int(native.mj_model.body_mocapid[goal])

    @jax.jit
    def render_goal(position):
        data = state.data.replace(mocap_pos=state.data.mocap_pos.at[:, mocap].set(position))
        # Leave native body/geom transforms stale: rendering must read the
        # current mocap pose without advancing or replacing native physics.
        rendered = native.render_state(state.replace(data=data))
        return rendered.obs['pixels/view_0'], rendered.data.geom_xpos

    assert mocap not in native._hidden_mocap_ids
    cube = native.mj_model.body('box').id
    near, native_positions = render_goal(state.data.xpos[:, cube])
    far, _ = render_goal(jnp.full((2, 3), 100.))
    assert not np.array_equal(near, far)
    np.testing.assert_array_equal(native_positions, state.data.geom_xpos)


def test_legacy_rgb_goal_visibility_preserves_native_physics():
    # Main's format-2 native-vision runs store target_pos in info rather than
    # moving the native mocap. Keep their visual correction and native rewards.
    config = env_config(DEFAULT_TASK, 'rgb')
    config.pop('rgb')
    config['environment']['vision'] = True
    visible = make_env(config, num_envs=2)
    env = wrapper.wrap_for_brax_training(visible, episode_length=2, action_repeat=1)
    values = ConfigDict(config['environment'])
    values.vision_config.nworld = 2
    original = registry.load(DEFAULT_TASK, config=values)
    official = wrapper.wrap_for_brax_training(original, episode_length=2, action_repeat=1)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    state, expected = jax.jit(env.reset)(keys), jax.jit(official.reset)(keys)
    step, native_step = jax.jit(env.step), jax.jit(official.step)
    for i in range(4):
        if i:
            state, expected = step(state, jnp.zeros((2, 3))), native_step(expected, jnp.zeros((2, 3)))
        assert not np.array_equal(state.obs['pixels/view_0'], expected.obs['pixels/view_0'])
        for name in ('qpos', 'qvel', 'ctrl', 'xpos', 'xmat', 'geom_xpos', 'geom_xmat', 'mocap_pos', 'mocap_quat'):
            np.testing.assert_array_equal(getattr(state.data, name), getattr(expected.data, name))
        np.testing.assert_array_equal(state.reward, expected.reward)
        np.testing.assert_array_equal(state.done, expected.done)
        np.testing.assert_array_equal(state.info['rng'], expected.info['rng'])
        for key in state.metrics:
            np.testing.assert_array_equal(state.metrics[key], expected.metrics[key])
    moved = state.replace(info={**state.info, 'target_pos': state.info['target_pos'] + jnp.array([0., .12, 0.])})
    moved = jax.jit(visible.render_state)(moved)
    assert not np.array_equal(moved.obs['pixels/view_0'], state.obs['pixels/view_0'])
    np.testing.assert_array_equal(moved.data.geom_xpos, state.data.geom_xpos)


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('mode', ['state', 'rgb', 'map'])
def test_training_checkpoint_and_evaluation(tmp_path, monkeypatch, mode, task):
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
        '--env-id', task, '--obs-mode', mode, '--num-envs', '2', '--num-eval-envs', '1',
        '--total-timesteps', '8', '--unroll-length', '2', '--batch-size', '2',
        '--num-minibatches', '1', '--num-updates-per-batch', '1', '--num-evals', '2',
        '--num-resets-per-eval', '1',
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
    assert policy.checkpoint.parent == run / 'checkpoints'
    env = make_env(policy.env_config, 1)
    env = wrapper.wrap_for_brax_training(env, policy.metadata['ppo']['episode_length'], policy.metadata['ppo']['action_repeat'])
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(7), 1))
    expected = jax.jit(inference[0])(state.obs, jax.random.PRNGKey(0))[0]
    np.testing.assert_allclose(policy.act(state.obs), expected, rtol=1e-6, atol=1e-6)
    summary, rows = evaluate(policy, episodes=3, num_envs=2)
    assert summary['episodes'] == len(rows) == 3
    assert all(np.isfinite(value) for row in rows for value in row.values())
    assert all(row['episode_len'] <= policy.metadata['ppo']['episode_length'] for row in rows)
    if mode == 'rgb' and task == DEFAULT_TASK:
        legacy = tmp_path / 'legacy-format2-rgb'
        shutil.copytree(run, legacy)
        metadata = json.loads((legacy / 'config.json').read_text())
        metadata['format_version'] = 2
        metadata['env_config'].pop('rgb')
        metadata['env_config']['environment']['vision'] = True
        metadata['env_config']['goal_markers'] = False
        metadata.pop('renderer', None)
        (legacy / 'config.json').write_text(json.dumps(metadata))
        restored = load_policy(legacy)
        native_legacy = make_env(restored.env_config, num_envs=1)
        assert native_legacy.unwrapped._vision
        assert not hasattr(native_legacy, 'render_metadata')
        summary, rows = evaluate(restored, episodes=1, num_envs=1)
        assert summary['episodes'] == len(rows) == 1
        assert np.isfinite(rows[0]['return'])
    if mode == 'map' and task == DEFAULT_TASK:
        moved = tmp_path / 'transferred-map-run'
        shutil.copytree(run, moved)
        bundle = moved / 'map-cache'
        bundle.mkdir()
        for cache_path in policy.metadata['map_cache_paths']:
            shutil.copy2(cache_path, bundle / Path(cache_path).name)
        metadata = json.loads((moved / 'config.json').read_text())
        metadata['env_config']['map']['dino'].update(source='/missing/dino-source', weights='/missing/weights.pth')
        metadata['map_cache_paths'] = ['/missing/original-cache/' + Path(path).name for path in metadata['map_cache_paths']]
        (moved / 'config.json').write_text(json.dumps(metadata))
        monkeypatch.setattr('benchmark.common.dino.FrozenDINO', lambda *_: pytest.fail('Cache hit must not load DINO'))
        portable = load_policy(moved)
        assert portable.checkpoint.parent == moved / 'checkpoints'
        assert all(Path(path).parent == bundle for path in portable.metadata['map_cache_paths'])
        summary, rows = evaluate(portable, episodes=1, num_envs=1)
        assert summary['episodes'] == len(rows) == 1


@pytest.mark.parametrize('mode', ['state', 'rgb', 'map'])
def test_real_checkpoint_continuation(tmp_path, monkeypatch, mode):
    train_module = importlib.import_module('benchmark.rl.train')
    real_save, real_train = train_module.ppo.checkpoint.save, train_module.ppo.train
    source = tmp_path / 'interrupted'

    def interrupt_after_save(*args, **kwargs):
        real_save(*args, **kwargs)
        raise InterruptedError('simulate worker eviction after finalized checkpoint')

    monkeypatch.setattr(train_module.ppo.checkpoint, 'save', interrupt_after_save)
    args = train_module.parser().parse_args([
        '--obs-mode', mode, '--num-envs', '2', '--num-eval-envs', '1', '--no-run-evals',
        '--total-timesteps', '16', '--unroll-length', '2', '--batch-size', '2',
        '--num-minibatches', '1', '--num-updates-per-batch', '1', '--checkpoint-steps', '8',
        '--num-resets-per-eval', '1', '--map-views', '4', '--map-extra-views', '0',
        '--output', str(source),
    ])
    with pytest.raises(InterruptedError, match='worker eviction'):
        train_module.train(args)
    checkpoint, saved_params = latest_checkpoint(source)
    assert int(checkpoint.name) == 8
    # Neither metrics after the save nor incomplete higher-number saves count.
    with open(source / 'train.csv', 'a') as stream:
        stream.write('999999,training/total_loss,0\n')
    (source / 'checkpoints' / '999999').mkdir()
    (source / 'checkpoints' / '1000000.orbax-checkpoint-tmp').mkdir()
    monkeypatch.setattr(train_module.ppo.checkpoint, 'save', real_save)
    initial = []

    def record(step, _make_policy, params):
        if step == 0:
            initial.append(jax.device_get(params))

    def observe_restore(**kwargs):
        assert kwargs['restore_value_fn']
        assert kwargs['num_timesteps'] == 8
        return real_train(**kwargs, policy_params_fn=record)

    monkeypatch.setattr(train_module.ppo, 'train', observe_restore)
    with pytest.warns(RuntimeWarning, match='Skipped invalid checkpoints'):
        resumed = train_module.train(train_module.parser().parse_args([
            '--resume', str(source), '--no-run-evals', '--checkpoint-steps', '8', '--output', str(tmp_path / 'resumed')]))
    assert len(initial) == 1
    for actual, expected in zip(jax.tree.leaves(initial[0]), jax.tree.leaves(saved_params)):
        np.testing.assert_array_equal(actual, expected)
    metadata = json.loads((resumed / 'config.json').read_text())
    assert (metadata['target_timesteps'], metadata['resume_base_steps'], metadata['ppo']['num_timesteps']) == (16, 8, 8)
    final_checkpoint, _ = latest_checkpoint(resumed)
    assert int(final_checkpoint.name) == 8
    monkeypatch.setattr(train_module.ppo, 'train', lambda **_: pytest.fail('Completed run retrained'))
    assert train_module.train(train_module.parser().parse_args(['--resume', str(resumed)])) == resumed
