"""Fast observation/network contracts, independent of robot assets and DINO weights."""

import copy
import json
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
import trimesh
from mujoco_playground import manipulation
from mujoco_playground.config import manipulation_params

import benchmark  # noqa: F401  Set runtime defaults before importing MuJoCo/JAX.
from benchmark.common.envs import (
    DEFAULT_TASK,
    GOAL_TASKS,
    OBS_MODES,
    TASKS,
    env_config,
    make_env,
    ppo_config,
)
from benchmark.common.mapping import (
    MAP_VERSION,
    FeatureBank,
    IsolatedRenderer,
    MapObservationWrapper,
    appearance_key,
    components,
    surface_points,
    visual_mesh,
)
from benchmark.common.points import MapEncoder
from benchmark.common.policy import network_factory
from benchmark.common.rgb import render_model, rgb_config
from benchmark.rl.train import parser


def test_native_configs_and_removed_modes(monkeypatch):
    assert set(TASKS) == set(manipulation.ALL_ENVS)
    assert OBS_MODES == ("state", "rgb", "map")
    for task in TASKS:
        config = env_config(task, "state")
        assert config["goal_markers"] == "task"
        assert config["environment"] == manipulation.get_default_config(task).to_dict()
        expected = manipulation_params.brax_ppo_config(task, config["environment"].get("impl")).to_dict()
        expected['num_updates_per_batch'] = 4
        assert ppo_config(config) == expected
    for mode in ("rgbd", "dino"):
        with pytest.raises(ValueError, match="Unsupported"):
            env_config(DEFAULT_TASK, mode)
    for task in TASKS:
        rgb = env_config(task, "rgb")
        native = env_config(task, "state")
        assert rgb['environment'] == native['environment']
        assert not rgb['environment'].get('vision', False)
        assert rgb['rgb']['resolution'] == [64, 64]
        assert rgb['rgb']['use_textures']
        expected = manipulation_params.brax_ppo_config(task, native['environment'].get('impl')).to_dict()
        expected.update(num_envs=128, num_eval_envs=8, batch_size=16, normalize_observations=False)
        expected['num_updates_per_batch'] = 4
        expected['network_factory'] = manipulation_params.brax_vision_ppo_config(DEFAULT_TASK).network_factory.to_dict()
        assert ppo_config(rgb) == expected
    for flag in ("--control-mode", "--view", "--state-input"):
        with pytest.raises(SystemExit):
            parser().parse_args([flag, "x"])
    assert parser().parse_args([]).map_background is None  # Resume inherits unspecified options.
    assert parser().parse_args([]).map_robot is None
    assert parser().parse_args(['--map-robot', 'gripper']).map_robot == 'gripper'
    with pytest.raises(SystemExit):
        parser().parse_args(['--map-robot', 'arm'])
    assert parser().parse_args(['--map-background', 'true']).map_background == 'true'
    with pytest.raises(SystemExit):
        parser().parse_args(['--map-background', 'table'])
    # Format-2 official Cartesian RGB must continue its old native vision path.
    rgb = env_config(DEFAULT_TASK, "rgb")
    assert GOAL_TASKS == set(TASKS) - {'AlohaSinglePegInsertion', 'LeapCubeRotateZAxis', 'AeroCubeRotateZAxis'}
    for mode in ('state', 'rgb', 'map'):
        with pytest.raises(ValueError, match='different goal visibility'):
            make_env({**env_config(DEFAULT_TASK, mode), 'goal_markers': False})
    del rgb['rgb']
    rgb['environment']['vision'] = True
    rgb['goal_markers'] = False
    monkeypatch.setattr('benchmark.common.envs.registry.load', lambda name, config: config)
    monkeypatch.setattr('benchmark.common.envs.hide_goal_markers', lambda *a, **kw: None)
    restored = make_env(json.loads(json.dumps(rgb)), num_envs=3)
    assert restored.vision_config.cam_res == (64, 64)
    assert restored.vision_config.nworld == 3
    assert restored.vision
    monkeypatch.setattr('benchmark.common.envs.GoalVisibleCartesian', lambda config: config)
    assert make_env({**rgb, 'goal_markers': 'task'}, num_envs=2).vision_config.nworld == 2
    with pytest.raises(ValueError, match='different goal visibility'):
        make_env({**rgb, 'goal_markers': True})
    params = ppo_config(env_config(DEFAULT_TASK, "map"))
    assert params['num_updates_per_batch'] == 4
    assert (params["num_envs"], params["batch_size"], params["normalize_observations"]) == (8, 1, False)


def test_added_camera_preserves_native_model_and_constructor_changes(tmp_path):
    xml = '''<mujoco><include file="robot.xml"/><statistic center=".4 .1 .2" extent=".8"/>
      <visual><global azimuth="150" elevation="-30"/></visual></mujoco>'''
    assets = {'robot.xml': b'''<mujoco><worldbody>
      <body name="robot" pos=".1 0 .2"><joint name="joint" type="hinge"/>
      <geom name="visual" type="box" size=".1 .2 .1"/></body></worldbody></mujoco>'''}
    source = tmp_path / 'scene.xml'
    source.write_text(xml)
    native = mujoco.MjModel.from_xml_string(xml, assets=assets)
    native.geom_rgba[0] = [.2, .8, .1, .9]
    native.body_pos[1] = [.2, .1, .3]
    before = copy.copy(native)
    env = SimpleNamespace(mj_model=native, xml_path=source, _model_assets=assets)
    rendered = render_model(env, rgb_config('PandaPickCube'))
    assert native.ncam == 0 and rendered.ncam == 1
    assert rendered.camera('benchmark_default').id == 0
    assert np.isfinite(rendered.cam_pos).all() and np.isfinite(rendered.cam_quat).all()
    for field in ('body_pos', 'geom_rgba', 'geom_size', 'jnt_axis', 'qpos0'):
        np.testing.assert_array_equal(getattr(native, field), getattr(before, field))
        np.testing.assert_array_equal(getattr(rendered, field), getattr(native, field))


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(2, 32, 3)).astype(np.float32)
    normals = rng.normal(size=xyz.shape).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    obs = {"xyz": jnp.asarray(xyz.reshape(2, -1)), "feature_ids": jnp.tile(jnp.arange(32), (2, 1)),
           'normals': jnp.asarray(normals.reshape(2, -1)), 'geometry_epsilon': jnp.full((2, 1), 1e-6)}
    bank = jnp.asarray(rng.normal(size=(32, 1024)), dtype=jnp.float32)
    net = MapEncoder()
    params = net.init(jax.random.PRNGKey(2), obs, bank)
    return net, params, obs, bank


def test_point_encoder_invariance_padding_and_gradients(points):
    net, params, obs, bank = points
    forward = jax.jit(net.apply)
    output = forward(params, obs, bank)
    assert output.shape == (2, 256)
    for block in ('block_0', 'block_1', 'global_block'):
        assert params['params'][block]['position_in']['kernel'].shape[0] == 7
    translated = {**obs, "xyz": (obs["xyz"].reshape(2, 32, 3) + jnp.array([1., 2., -.5])).reshape(2, -1)}
    np.testing.assert_allclose(forward(params, translated, bank), output, atol=2e-5, rtol=2e-5)
    padded = {**obs, "xyz": jnp.pad(obs["xyz"].reshape(2, 32, 3), ((0, 0), (0, 5), (0, 0))).reshape(2, -1),
              'normals': jnp.pad(obs['normals'].reshape(2, 32, 3), ((0, 0), (0, 5), (0, 0))).reshape(2, -1),
              "feature_ids": jnp.pad(obs["feature_ids"], ((0, 0), (0, 5)), constant_values=-1)}
    np.testing.assert_allclose(forward(params, padded, bank), output, atol=2e-5, rtol=2e-5)
    grad = jax.jit(jax.grad(lambda p: jnp.square(net.apply(p, obs, bank)).mean()))(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert any(np.any(x != 0) for x in jax.tree.leaves(grad))


def test_map_networks_ignore_normalizer_and_have_separate_encoders(points):
    _, _, obs, bank = points
    factory = network_factory("map", {}, SimpleNamespace(features=bank))
    nets = factory({k: v.shape[1:] for k, v in obs.items()}, 3,
                   preprocess_observations_fn=lambda *_: pytest.fail("Map must not normalize xyz/IDs"))
    actor = nets.policy_network.init(jax.random.PRNGKey(1))
    critic = nets.value_network.init(jax.random.PRNGKey(2))
    assert nets.policy_network.apply(None, actor, obs).shape == (2, 6)
    assert nets.value_network.apply(None, critic, obs).shape == (2,)
    assert not np.array_equal(actor['params']['MapEncoder_0']['input']['kernel'],
                              critic['params']['MapEncoder_0']['input']['kernel'])


def test_surface_voxels_and_normals():
    mesh = trimesh.creation.box(extents=(.1, .1, .1))
    xyz, normals = surface_points(mesh, .015)
    assert len(np.unique(np.floor(xyz / .015), axis=0)) == len(xyz)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)


def test_background_selection_keeps_all_robot_and_articulated_parts():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor_box" type="box" size="1 1 .1"/>
      <geom name="wall_mesh_proxy" type="box" size=".1 1 1"/>
      <geom name="plain_plane" type="plane" size="1 1 .1"/>
      <geom name="camera_housing" size=".02"/>
      <body name="link0"><geom name="base" size=".1"/>
        <body name="hand"><geom name="hand_visual" size=".03"/>
          <body name="finger"><joint/><geom name="finger_visual" size=".01"/></body>
        </body>
      </body>
      <body name="handle"><joint type="slide"/><geom name="handle_visual" size=".02"/>
        <body name="drawer_child"><geom name="drawer_part" size=".03"/></body>
      </body>
      <body name="furniture"><geom name="support" size=".04"/></body>
      <body name="mocap_target" mocap="true"><geom name="goal" size=".01"/></body>
    </worldbody></mujoco>''')
    def selected(background, robot='full'):
        return {model.geom(g).name for _, geoms in components(model, 'PandaOpenCabinet',
                                                                {'background': background, 'robot': robot}) for g in geoms}
    assert selected(False) == {'base', 'hand_visual', 'finger_visual', 'handle_visual', 'drawer_part', 'goal'}
    assert selected(True) == selected(False) | {'camera_housing', 'support'}
    assert selected(False, 'gripper') == selected(False) - {'base'}
    assert selected(True, 'gripper') == selected(True) - {'base'}
    with pytest.raises(ValueError, match='background must be true or false'):
        selected('none')
    with pytest.raises(ValueError, match='robot must be full or gripper'):
        selected(False, 'arm')


@pytest.mark.parametrize('kind,size', [('box', '.02 .03 .04'), ('sphere', '.02'),
                                     ('ellipsoid', '.02 .03 .04'), ('capsule', '.02 .03'),
                                     ('cylinder', '.02 .03')])
def test_primitive_visual_meshes(kind, size):
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><geom type="{kind}" size="{size}" pos=".1 .2 .3"/></worldbody></mujoco>')
    mesh = visual_mesh(model, [0])
    assert len(mesh.faces) and np.isfinite(mesh.vertices).all()
    np.testing.assert_allclose(mesh.bounds.mean(0), [.1, .2, .3], atol=1e-6)


def test_cache_key_tracks_appearance_and_extraction():
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <texture name="tex" type="2d" builtin="checker" width="16" height="16"/>
      <material name="mat" texture="tex"/></asset><worldbody><body>
      <geom type="box" size=".02 .02 .03" material="mat"/></body></worldbody></mujoco>''')
    config = {'voxel_size': .015, 'views': 96, 'extra_views': 512,
              'dino': {'sha256': 'weights-a', 'source_revision': 'source-a'}}

    def key(m=model, c=config):
        return appearance_key(m, [0], visual_mesh(m, [0]), c)

    original = key()
    model.body_pos[1] += 1  # Runtime placement is intentionally independent of local maps.
    assert key() == original
    for field in ('geom_size', 'mat_rgba', 'tex_data'):
        changed = copy.copy(model)
        getattr(changed, field).flat[0] += 1
        assert key(changed) != original
    assert key(c={**config, 'views': 4}) != original
    assert key(c={**config, 'dino': {**config['dino'], 'sha256': 'weights-b'}}) != original


def test_feature_bank_validation_and_reuse(tmp_path):
    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = MAP_VERSION
        f["xyz"] = np.zeros((2, 3), np.float32)
        f['normals'] = np.tile([0., 0., 1.], (2, 1)).astype(np.float32)
        f["features"] = np.ones((2, 1024), np.float16)
    bank = FeatureBank()
    a = bank.add(path)
    b = bank.add(path)
    assert a is b and bank.size == 2
    assert bank.features.shape == (2, 1024)
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="Incompatible"):
        FeatureBank().add(path)


def test_articulated_parts_follow_forward_kinematics():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base" pos="1 2 3"><geom size=".1"/>
        <body name="arm" pos=".3 0 0"><joint axis="0 0 1"/>
          <geom size=".1" pos=".1 0 0"/>
          <body name="finger" pos=".2 0 0"><joint type="slide" axis="1 0 0"/>
            <geom size=".02" pos=".02 .01 0"/>
          </body>
        </body>
      </body>
    </worldbody></mujoco>''')
    mapper = object.__new__(MapObservationWrapper)
    mapper.local = jnp.array([[0, 0, 0], [.1, 0, 0], [.02, .01, 0]])
    mapper.body_ids = jnp.array([1, 2, 3])
    mapper.feature_ids = jnp.arange(3)
    mapper.local_normals = jnp.tile(jnp.array([1., 0, 0]), (3, 1))
    mapper.frame_bodies, mapper.frame_axes = jnp.array([0]), 0
    mapper.mocap_points = jnp.array([], dtype=jnp.int32)
    data = mujoco.MjData(model)
    for qpos, expected in (
        ([0, 0], [[1, 2, 3], [1.4, 2, 3], [1.52, 2.01, 3]]),
        ([np.pi / 2, .04], [[1, 2, 3], [1.3, 2.1, 3], [1.29, 2.26, 3]]),
    ):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        poses = SimpleNamespace(xmat=jnp.asarray(data.xmat.reshape(-1, 3, 3)), xpos=jnp.asarray(data.xpos))
        out, _ = mapper.geometry(poses)
        np.testing.assert_allclose(out, expected, atol=1e-6)


def test_goal_map_uses_current_mocap_pose_without_advancing_physics():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object"><geom size=".02"/></body>
      <body name="mocap_target" mocap="true"><geom type="box" size=".02 .03 .04"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mapper = object.__new__(MapObservationWrapper)
    mapper.local = jnp.array([[.02, 0, 0], [.02, 0, 0], [0, .03, 0]])
    mapper.body_ids = jnp.array([1, 2, 2])
    mapper.feature_ids = jnp.arange(3)
    mapper.local_normals = jnp.tile(jnp.array([1., 0, 0]), (3, 1))
    mapper.frame_bodies, mapper.frame_axes = jnp.array([0]), 0
    mapper.mocap_points = jnp.array([1, 2])
    mapper.mocap_ids = jnp.array([0, 0])
    # A native task can change mocap after stepping, leaving body transforms stale.
    poses = SimpleNamespace(xpos=jnp.asarray(data.xpos), xmat=jnp.asarray(data.xmat.reshape(-1, 3, 3)),
                            mocap_pos=jnp.array([[1., 2., 3.]]),
                            mocap_quat=jnp.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]]))
    xyz, normals = mapper.geometry(poses)
    np.testing.assert_allclose(xyz, [[.02, 0, 0], [1, 2.02, 3], [.97, 2, 3]], atol=1e-6)
    np.testing.assert_array_equal(poses.xpos, data.xpos)
    np.testing.assert_allclose(normals, [[1, 0, 0], [0, 1, 0], [0, 1, 0]], atol=1e-6)


@pytest.mark.integration
def test_isolated_renderer_geometry_depth_and_model_preservation():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="box" pos="1 2 3"><geom type="box" size=".02 .02 .03" pos=".01 0 0" rgba="1 0 0 1"/></body>
      <geom type="box" size=".5 .5 .5" rgba="0 1 0 1"/>
      <site name="a" pos="-.03 0 .1"/><site name="b" pos=".03 0 .1"/>
    </worldbody><tendon><spatial width=".004" rgba="0 1 0 1">
      <site site="a"/><site site="b"/>
    </spatial></tendon></mujoco>''')
    before = copy.copy(model)
    geom = int(model.body_geomadr[model.body("box").id])
    mesh = visual_mesh(model, [geom])
    np.testing.assert_allclose(mesh.bounds, [[-.01, -.02, -.03], [.03, .02, .03]])
    renderer = IsolatedRenderer(model, [geom])
    try:
        rgb, depth, pos, rot, _ = renderer.render(np.array([.01, 0, .2]), np.array([.01, 0, 0]))
        assert rgb[128, 128, 0] > rgb[128, 128, 1]
        np.testing.assert_allclose(depth[128, 128], .17, atol=.001)
        np.testing.assert_allclose(pos, [.01, 0, .2], atol=1e-6)
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-6)
    finally:
        renderer.close()
    for name in ("geom_group", "geom_pos", "geom_rgba", "body_pos"):
        np.testing.assert_array_equal(getattr(model, name), getattr(before, name))
