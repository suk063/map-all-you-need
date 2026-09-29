"""Real task selection, background caches and browser scene construction."""

import gc

import jax
import mujoco
import numpy as np
import pytest

from benchmark.common.envs import GOAL_BODIES, GOAL_TASKS, TASKS, env_config, make_env
from benchmark.common.mapping import MapObservationWrapper, components
from tests.test_integration import map_settings
from util.view_scene import load_scene

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('task', TASKS)
def test_scene_selection_and_viewer(task):
    from mjviser import ViserMujocoScene
    import viser

    model, data, reset = load_scene(task, seed=7)
    goals = np.array([model.body(int(b)).name in GOAL_BODIES for b in model.geom_bodyid])
    if task not in GOAL_TASKS:
        assert np.all(model.geom_rgba[goals, 3] == 0)
    selections = [{g for _, geoms in components(model, task, {'background': background}) for g in geoms}
                  for background in (False, True)]
    core, background = selections
    assert core <= background
    assert all(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE for g in background)
    if task.startswith('Aloha'):
        meshes = [{model.mesh(int(model.geom_dataid[g])).name for g in ids
                   if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH} for ids in selections]
        assert all({'tabletop', 'tablelegs'}.isdisjoint(meshes_) for meshes_ in meshes)
        assert 'd405_solid' in meshes[1]
        assert len(background - core) > 2
    elif task == 'PandaRobotiqPushCube':
        assert {model.geom(g).name for g in background - core} == {'pad', 'camera_tracking_box'}
    else:
        assert core == background
    gripper = {g for _, geoms in components(model, task, {'background': False, 'robot': 'gripper'}) for g in geoms}
    gripper_bg = {g for _, geoms in components(model, task, {'background': True, 'robot': 'gripper'}) for g in geoms}
    assert gripper < core
    assert gripper_bg - gripper == background - core
    names = {model.body(int(model.geom_bodyid[g])).name for g in gripper}
    if task.startswith('Aloha'):
        assert {'left/gripper_base', 'left/left_finger_link', 'right/right_finger_link'} <= names
        assert {'left/base_link', 'right/wrist_link'}.isdisjoint(names)
    elif task == 'PandaRobotiqPushCube':
        # Pad bodies contain only collision proxies; visible fingers live on followers.
        assert {'base', 'left_follower', 'right_follower', 'box'} <= names
        assert {'link0', 'link7', 'fts300_body'}.isdisjoint(names)
    elif task.startswith('Panda'):
        assert names == {'hand', 'left_finger', 'right_finger', 'mocap_target',
                         'handle' if task == 'PandaOpenCabinet' else 'box'}
    else:
        assert {'palm', 'cube'} <= names
        assert {'leap_mount', 'tetheria_mount'}.isdisjoint(names)
    before = data.qpos.copy()
    server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
    try:
        scene = ViserMujocoScene(server, model, num_envs=1)
        scene.update_from_mjdata(data)
        mujoco.mj_step(model, data)
        scene.update_from_mjdata(data)
        reset(model, data)
        np.testing.assert_array_equal(data.qpos, before)
        assert np.isfinite(data.xpos).all()
    finally:
        server.stop()
        jax.clear_caches()
        gc.collect()


@pytest.mark.parametrize('task', ['AlohaHandOver', 'PandaRobotiqPushCube'])
def test_background_cache_and_map_view(task, monkeypatch):
    config = {**map_settings(), 'background': True, 'robot': 'gripper'}
    env = make_env(env_config(task))
    mapped = MapObservationWrapper(env, task, config)
    with monkeypatch.context() as patch:
        patch.setattr('benchmark.common.mapping.build_template', lambda *a, **kw: pytest.fail('Cache miss'))
        cached = MapObservationWrapper(env, task, config)
        assert mapped.bank.paths == cached.bank.paths
    model, data, _ = load_scene(task, map_only=True, background=True, robot='gripper')
    selected = {g for _, geoms in components(env.mj_model, task, config) for g in geoms}
    visible = set(np.flatnonzero(model.geom_rgba[:, 3] > 0))
    assert visible == selected
    np.testing.assert_array_equal(model.geom_contype, env.mj_model.geom_contype)
    np.testing.assert_array_equal(model.geom_conaffinity, env.mj_model.geom_conaffinity)
    assert np.isfinite(data.qpos).all()
    jax.clear_caches()
    gc.collect()


def test_dino_pca_uses_training_cache_before_map_only_filtering(monkeypatch):
    from util.dino_pca import pca_colors

    task = 'PandaPickCubeCartesian'
    env = make_env(env_config(task))
    config = {**map_settings(), 'robot': 'gripper'}
    mapped = MapObservationWrapper(env, task, config)
    # The scene reuses this native environment; filtering changes its private
    # render copy only, so training and visualization still share cache keys.
    original_rgba = env.mj_model.geom_rgba.copy()
    model, data, reset = load_scene(task, map_only=True, robot='gripper', env=env)
    np.testing.assert_array_equal(env.mj_model.geom_rgba, original_rgba)
    assert np.count_nonzero(model.geom_rgba[:, 3]) < np.count_nonzero(original_rgba[:, 3])
    with monkeypatch.context() as patch:
        patch.setattr('benchmark.common.mapping.build_template', lambda *a, **kw: pytest.fail('Cache miss'))
        cached = MapObservationWrapper(env, task, config)
    assert cached.bank.paths == mapped.bank.paths
    features = np.concatenate(cached.bank.arrays)[np.asarray(cached.feature_ids)]
    colors = pca_colors(features)
    assert colors.shape == (len(cached.local), 3) and np.ptp(colors) > 0
    initial = data.qpos.copy()
    mujoco.mj_step(model, data)
    reset(model, data)
    np.testing.assert_array_equal(data.qpos, initial)
    jax.clear_caches()
    gc.collect()
