"""Scene-wide DINO PCA colors and a body-local point-cloud comparison."""

import numpy as np


def pca_embedding(features):
    """Return three centered PC scores, percentile-scaled RGB, and variance ratios."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or min(features.shape) < 1 or not np.isfinite(features).all():
        raise ValueError("PCA features must be a nonempty, finite (points, channels) array")
    centered = features - features.mean(axis=0)
    colors = np.full((len(features), 3), .5)
    scores = np.zeros((len(features), 3))
    ratios = np.zeros(3)
    # The covariance matrix is only 1024 x 1024 for DINOv3 ViT-L, independent
    # of scene size. Do not compute the much larger left singular vectors.
    values, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1][:3]
    for channel, index in enumerate(order):
        if values[index] <= max(values[-1], 0.) * 1e-10:
            continue  # Missing/constant PCs stay neutral, including rank-one maps.
        axis = vectors[:, index]
        axis = axis * (1 if axis[np.argmax(np.abs(axis))] >= 0 else -1)
        projected = centered @ axis
        scores[:, channel] = projected
        ratios[channel] = values[index] / np.maximum(values, 0).sum()
        low, high = np.percentile(projected, (2, 98))
        if high > low:
            colors[:, channel] = np.clip((projected - low) / (high - low), 0, 1)
    return scores, np.rint(colors * 255).astype(np.uint8), ratios


def pca_colors(features):
    """Map the first three centered PCs to RGB, fitting once for the whole scene."""
    return pca_embedding(features)[1]


class PCAView:
    """Fixed local points/colors; only rigid body poses change during playback."""

    def __init__(self, server, local, body_ids, colors, data, offset, point_size=.015):
        self.server = server
        self.local = np.asarray(local, dtype=np.float32)
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.root = server.scene.add_frame('/dino_pca', show_axes=False, position=self.offset)
        self.parts = {}
        for body in np.unique(self.body_ids):
            selected = self.body_ids == body
            name = f'/dino_pca/body_{body}'
            frame = server.scene.add_frame(name, show_axes=False)
            cloud = server.scene.add_point_cloud(
                f'{name}/points', points=self.local[selected], colors=colors[selected],
                point_size=point_size, point_shape='circle', precision='float32')
            self.parts[int(body)] = (frame, cloud)
        world = self.world_points(data)
        low, high = world.min(0), world.max(0)
        self.label_position = (low + high) / 2
        self.label_position[2] = high[2] + max(.08, np.max(high - low) * .1)
        self.scene_label = server.scene.add_label('/scene_label', 'Scene', position=self.label_position)
        server.scene.add_label('/dino_pca/label', 'DINO embedding · PCA → RGB', position=self.label_position)
        self.update(data)

    def world_points(self, data):
        rotation = data.xmat[self.body_ids].reshape(-1, 3, 3)
        return np.einsum('nij,nj->ni', rotation, self.local) + data.xpos[self.body_ids]

    def update(self, data, scene_offset=(0., 0., 0.)):
        with self.server.atomic():
            self.root.position = self.offset + scene_offset
            self.scene_label.position = self.label_position + scene_offset
            for body, (frame, _) in self.parts.items():
                frame.position = data.xpos[body].copy()
                frame.wxyz = data.xquat[body].copy()

    def add_gui(self, data):
        with self.server.gui.add_folder('DINO PCA'):
            self.server.gui.add_markdown(
                f'**{len(self.local):,} map points** · PC1 / PC2 / PC3 → R / G / B.\n\n'
                'One PCA for this scene; colors stay fixed while bodies move.')
            size = self.server.gui.add_slider('Point size', min=.001, max=.05, step=.001, initial_value=.015)

            @size.on_update
            def _(_event):
                for _, cloud in self.parts.values():
                    cloud.point_size = size.value

            frame = self.server.gui.add_button('Frame comparison')

            @frame.on_click
            def _(event):
                if event.client is None:
                    return
                world = self.world_points(data)
                center = (world.min(0) + world.max(0) + self.offset) / 2
                center += self.root.position - self.offset  # Native camera tracking.
                distance = max(np.linalg.norm(np.ptp(world, axis=0) + np.abs(self.offset)) * 1.8, .5)
                event.client.camera.look_at = center
                event.client.camera.position = center + np.array([.470, -.814, .342]) * distance
                event.client.camera.up_direction = (0., 0., 1.)
