"""Contact geometry properties; no performance comparisons or training runs."""

from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from benchmark.common.geometry import (
    EPSILON_M,
    geometry_config,
    scene_bounds,
    validate_geometry,
)
from benchmark.common.mapping import MAP_VERSION, FeatureBank, MapObservationWrapper
from benchmark.common.points import MapEncoder, contact_features


def test_contact_pairs_and_safe_derivatives():
    delta = jnp.array([[1., 0, 0], [0, .2, 0], [0, 0, 0], [1e-8, 0, 0]])
    a = jnp.tile(jnp.array([1., 0, 0]), (4, 1))
    b = a.at[0].set(jnp.array([-1., 0, 0]))
    edge = contact_features(delta, a, b, EPSILON_M)
    np.testing.assert_allclose(edge[0], [1, 0, 0, 1, -1, 1, -1])
    np.testing.assert_allclose(edge[1], [0, .2, 0, .2, 1, 0, 0])
    np.testing.assert_array_equal(edge[2], [0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(edge[3, -2:], [0, 0])
    grads = jax.grad(lambda d: contact_features(d, a, b, EPSILON_M).sum())(delta)
    assert np.isfinite(grads).all()
    # Initialization can have zero epsilon and duplicate dummy coordinates.
    assert np.isfinite(contact_features(jnp.zeros_like(delta), a, b, 0)).all()
    ordinary = contact_features(delta[:2], a[:2], b[:2], EPSILON_M)
    np.testing.assert_array_equal(ordinary, contact_features(delta[:2], a[:2], b[:2], .01))


def test_isotropic_bounds_preserve_contact_angles_and_distance_ratios():
    xyz = jnp.array([[0., 0, 0], [4, 2, 1], [1, 1, 0], [1e8, 1e8, 1e8]])
    valid = jnp.array([True, True, True, False])
    center, scale = jax.jit(scene_bounds)(xyz, valid)
    np.testing.assert_allclose(center, [2, 1, .5])
    np.testing.assert_allclose(scale, [2])
    normalized = (xyz - center) / scale
    assert np.abs(normalized[:3]).max() == 1
    normal = jnp.tile(jnp.array([0., 0, 1]), (2, 1))
    before = contact_features(xyz[1:3] - xyz[:1], normal, -normal, EPSILON_M)
    after = contact_features(normalized[1:3] - normalized[:1], normal, -normal, EPSILON_M / scale)
    np.testing.assert_allclose(after[:, :4], before[:, :4] / scale)
    np.testing.assert_allclose(after[:, 4:], before[:, 4:])
    np.testing.assert_allclose(after[0, 3] / after[1, 3], before[0, 3] / before[1, 3])


@pytest.mark.parametrize('valid', [jnp.array([True, True]), jnp.array([False, False])])
def test_degenerate_bounds_are_finite(valid):
    xyz = jnp.zeros((2, 3))
    center, scale = scene_bounds(xyz, valid)
    np.testing.assert_array_equal(center, [0, 0, 0])
    np.testing.assert_allclose(scale, [EPSILON_M])
    assert np.isfinite((xyz - center) / scale).all()


def mapper_and_poses():
    mapper = object.__new__(MapObservationWrapper)
    mapper.local = jnp.array([[-1., 0, 0], [1., 0, 0], [0, .5, 0]])
    mapper.local_normals = jnp.tile(jnp.array([0., 0, 1]), (3, 1))
    mapper.feature_ids = jnp.arange(3)
    mapper.body_ids = jnp.full((3,), 3)
    mapper.mocap_points = jnp.array([], dtype=jnp.int32)
    mapper.frame_bodies, mapper.frame_axes = jnp.array([1, 2]), 1
    poses = SimpleNamespace(xpos=jnp.array([[0., 0, 0], [-2., 1, 0], [2., 1, 0], [0, 2., 0]]),
                            xmat=jnp.tile(jnp.eye(3), (4, 1, 1)))
    return mapper, poses


def test_robot_frame_and_encoder_are_invariant_to_common_rigid_transform():
    mapper, poses = mapper_and_poses()
    xyz, normals = mapper.geometry(poses)
    np.testing.assert_allclose(xyz, [[-1, 1, 0], [1, 1, 0], [0, 1.5, 0]])
    rotation = jnp.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
    transformed = SimpleNamespace(xpos=poses.xpos @ rotation.T + jnp.array([3., -2, 7]),
                                  xmat=jnp.einsum('ij,njk->nik', rotation, poses.xmat))
    moved_xyz, moved_normals = mapper.geometry(transformed)
    np.testing.assert_allclose(moved_xyz, xyz, atol=1e-6)
    np.testing.assert_allclose(moved_normals, normals, atol=1e-6)
    center, scale = scene_bounds(xyz, mapper.feature_ids >= 0)
    info = {'_map_center': center, '_map_scale': scale}
    obs, moved = mapper.observation(poses, info), mapper.observation(transformed, info)
    bank = jax.random.normal(jax.random.PRNGKey(0), (3, 1024))
    net = MapEncoder()
    params = net.init(jax.random.PRNGKey(1), obs, bank)
    np.testing.assert_allclose(net.apply(params, obs, bank), net.apply(params, moved, bank), atol=2e-5)
    # Aloha's right arm orientation must not define/average the shared axes.
    turned_right = SimpleNamespace(xpos=poses.xpos, xmat=poses.xmat.at[2].set(rotation))
    np.testing.assert_allclose(mapper.geometry(turned_right)[0], xyz)
    assert geometry_config('AlohaHandOver')['frame'] == {
        'origin_bodies': ['left/base_link', 'right/base_link'], 'axes_body': 'left/base_link'}


@struct.dataclass
class State:
    data: object
    obs: object
    info: dict


def test_episode_bounds_are_fixed_without_clipping_and_explicit_reset_recomputes():
    mapper, initial = mapper_and_poses()
    poses = [initial]
    mapper.env = SimpleNamespace(
        reset=lambda _: State(poses[0], jnp.zeros(2), {}),
        step=lambda state, _: state.replace(data=poses[0], obs=state.obs + 1))
    state = mapper.reset(None)
    center, scale = state.info['_map_center'], state.info['_map_scale']
    poses[0] = SimpleNamespace(xmat=initial.xmat, xpos=initial.xpos.at[3, 0].add(10))
    stepped = mapper.step(state, None)
    np.testing.assert_array_equal(stepped.info['_map_center'], center)
    np.testing.assert_array_equal(stepped.info['_map_scale'], scale)
    assert stepped.obs['xyz'].max() > 1  # Never clip or recompute from moving geometry.
    reset = mapper.reset(None)
    assert np.abs(reset.obs['xyz']).max() <= 1
    assert not np.array_equal(reset.info['_map_center'], center)
    np.testing.assert_allclose(state.obs['geometry_epsilon'], EPSILON_M / scale)


@pytest.mark.parametrize('normal', [None, [[0., 0, 0]], [[0., 0, 2]], [[float('nan'), 0, 1]]])
def test_missing_invalid_normals_rejected(tmp_path, normal):
    path = tmp_path / 'map.h5'
    with h5py.File(path, 'w') as cache:
        cache.attrs['format_version'] = MAP_VERSION
        cache['xyz'] = np.zeros((1, 3))
        cache['features'] = np.zeros((1, 1024))
        if normal is not None:
            cache['normals'] = normal
    with pytest.raises(ValueError, match='Missing|Nonunit|Nonfinite'):
        FeatureBank().add(path)


def test_geometry_metadata_must_match_recipe():
    config = {'env_id': 'PandaPickCube', 'obs_mode': 'map', 'map_geometry': geometry_config('PandaPickCube')}
    validate_geometry(config)
    config['map_geometry']['normalization']['clip'] = True
    with pytest.raises(ValueError, match='retrain'):
        validate_geometry(config)
