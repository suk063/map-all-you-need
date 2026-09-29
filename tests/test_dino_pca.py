"""DINO PCA colors and rigid-body display alignment."""

import numpy as np
import pytest


def test_pca_colors_follow_principal_axes_and_preserve_equal_features():
    from util.dino_pca import pca_colors, pca_embedding

    # Orthogonal, unequal-variance factors, mixed into correlated features.
    factors = np.array([[a, b, c] for a in (-4., 4.) for b in (-2., 2.) for c in (-1., 1.)])
    mixing = np.array([[1., 1., 0., 0.], [-1., 1., 0., 0.], [0., 0., 1., 1.]]) / np.sqrt(2)
    features = factors @ mixing + np.array([17., -3., 9., 2.])
    scores, _, ratios = pca_embedding(features)
    np.testing.assert_allclose(scores.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(ratios, np.array([16., 4., 1.]) / 21)
    np.testing.assert_allclose(np.var(scores, axis=0), [16., 4., 1.])
    colors = pca_colors(np.concatenate([features, features]))
    assert colors.dtype == np.uint8
    assert colors.shape == (16, 3)
    np.testing.assert_array_equal(colors[:8], colors[8:])
    for channel in range(3):
        assert abs(np.corrcoef(colors[:8, channel], factors[:, channel])[0, 1]) > .999
    np.testing.assert_array_equal(pca_colors(features), pca_colors(features + 32))
    np.testing.assert_array_equal(pca_colors(features), pca_colors(features * 1e-8))
    order = np.array([5, 3, 1, 7, 0, 6, 4, 2])
    np.testing.assert_array_equal(pca_colors(features[order]), pca_colors(features)[order])


@pytest.mark.parametrize('features', [np.ones((1, 1024)), np.ones((5, 4)) * 7])
def test_constant_features_have_neutral_colors(features):
    from util.dino_pca import pca_colors

    np.testing.assert_array_equal(pca_colors(features), np.full((len(features), 3), 128, np.uint8))


def test_rank_one_features_do_not_amplify_numerical_noise():
    from util.dino_pca import pca_colors

    features = np.arange(9)[:, None] * np.array([[1., 2., 3., 4.]])
    colors = pca_colors(features)
    assert colors[0, 0] == 0 and colors[-1, 0] == 255
    np.testing.assert_array_equal(colors[:, 1:], np.full((9, 2), 128, np.uint8))


@pytest.mark.parametrize('features', [np.empty((0, 4)), np.ones(4), np.empty((2, 0)),
                                    np.array([[1., np.nan]]), np.array([[np.inf, 0.]])])
def test_pca_rejects_invalid_features(features):
    from util.dino_pca import pca_colors

    with pytest.raises(ValueError, match='features'):
        pca_colors(features)


def test_pca_cloud_follows_body_pose_tracking_and_reset():
    import mujoco
    viser = pytest.importorskip('viser')
    from util.dino_pca import PCAView

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="moving" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    local = np.array([[.1, 0, 0], [0, .1, 0], [0, 0, .2]], np.float32)
    bodies = np.array([1, 1, 0])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    offset = np.array([2., 0, 0])
    server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
    try:
        view = PCAView(server, local, bodies, colors, data, offset=offset)
        for move in (True, False):
            if move:
                data.qpos[:3] = [.5, -.2, 1.2]
                data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
            else:
                mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            tracking = -data.xpos[1] if move else np.zeros(3)
            view.update(data, tracking)
            for body, (frame, cloud) in view.parts.items():
                selected = bodies == body
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, frame.wxyz)
                displayed = cloud.points @ rotation.reshape(3, 3).T + frame.position + view.root.position
                expected = local[selected] @ data.xmat[body].reshape(3, 3).T + data.xpos[body] + offset + tracking
                np.testing.assert_allclose(displayed, expected, atol=1e-6)
                np.testing.assert_array_equal(cloud.points, local[selected])
                np.testing.assert_array_equal(cloud.colors, colors[selected])
            np.testing.assert_allclose(view.scene_label.position, view.label_position + tracking)
    finally:
        server.stop()


def test_viewer_pca_cli_keeps_existing_defaults():
    from util.view_scene import parser

    defaults = parser().parse_args([])
    assert not defaults.dino_pca and not defaults.map_only
    assert (defaults.map_views, defaults.map_extra_views) == (96, 512)
    args = parser().parse_args(['--dino-pca', '--map-only', '--map-robot', 'gripper',
                               '--map-cache', '/tmp/pca-maps', '--map-views', '4',
                               '--map-extra-views', '0', '--dino-source', '/tmp/dino',
                               '--dino-weights', '/tmp/dino.pth'])
    assert args.dino_pca and args.map_only and args.map_robot == 'gripper'
    assert (args.map_views, args.map_extra_views) == (4, 0)
    assert args.dino_source == '/tmp/dino' and args.dino_weights == '/tmp/dino.pth'
