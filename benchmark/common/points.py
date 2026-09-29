"""Flax port of the relative-only SERF-VLA Point Transformer.

The relative Point Transformer in `benchmark/common/points.py` is adapted from
`ExistentialRobotics/SERF-VLA`, commit `ea27b7aa753cf7da6def975846ccb5d3180e46f7`,
`src/serf_b1k/models/point_transformer_local.py`.

MIT License

Copyright (c) 2026 Byeonghyun Pak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

Dense = partial(nn.Dense, precision=jax.lax.Precision.HIGHEST)


def gather(x, ids):
    return jax.vmap(lambda a, i: a[i])(x, ids)


def sampling_coordinates(xyz, origin):
    return jnp.round((xyz - origin) * 10000)


def fps(xyz, valid, limit):
    xyz = sampling_coordinates(xyz, xyz[:, :1])
    size = min(limit, xyz.shape[1])
    first = jnp.argmax(valid, axis=1)

    def select(carry, _):
        distance, farthest = carry
        center = gather(xyz, farthest[:, None])
        distance = jnp.minimum(distance, jnp.sum((xyz - center) ** 2, axis=-1))
        next_id = jnp.argmax(jnp.where(valid, distance, -1), axis=1)
        return (distance, next_id), farthest

    _, ids = jax.lax.scan(select, (jnp.full(valid.shape, jnp.inf), first), None, length=size)
    return ids.T, jnp.arange(size)[None] < valid.sum(1)[:, None]


def knn(xyz, centers, valid, k):
    origin = xyz[:, :1]
    xyz = sampling_coordinates(xyz, origin)
    centers = sampling_coordinates(centers, origin)
    k = min(k, xyz.shape[1])

    def neighbors(center):
        distance = jnp.sum((center[:, None] - xyz) ** 2, axis=-1)
        return jax.lax.top_k(-jnp.where(valid, distance, jnp.inf), k)[1]

    # Bound temporary distance storage without changing the input point set.
    return jax.lax.map(neighbors, centers.swapaxes(0, 1), batch_size=32).swapaxes(0, 1)


class PointBlock(nn.Module):
    width: int
    neighbors: int = 16

    @nn.compact
    def __call__(self, xyz, features, valid):
        x = Dense(self.width, use_bias=False, name="before_dense")(features)
        x = nn.relu(nn.RMSNorm(epsilon=1e-6, name="before_norm")(x))
        q, k, v = jnp.split(Dense(3 * self.width, name="qkv")(x), 3, axis=-1)
        ids = knn(xyz, xyz, valid, self.neighbors)
        delta = Dense(self.width, name="position_in")(gather(xyz, ids) - xyz[:, :, None])
        delta = nn.relu(nn.RMSNorm(epsilon=1e-6, name="position_norm")(delta))
        delta = Dense(self.width, name="position_out")(delta)
        scores = nn.relu(nn.RMSNorm(epsilon=1e-6, name="weight_norm1")(gather(k, ids) - q[:, :, None] + delta))
        scores = Dense(self.width // 8, name="weight_in")(scores)
        scores = nn.relu(nn.RMSNorm(epsilon=1e-6, name="weight_norm2")(scores))
        scores = Dense(self.width // 8, name="weight_out")(scores)
        scores = jnp.where(gather(valid, ids)[..., None], scores, -jnp.finfo(scores.dtype).max)
        content = gather(v, ids) + delta
        content = content.reshape(*content.shape[:-1], 8, self.width // 8)
        x = (jax.nn.softmax(scores, axis=2)[..., None, :] * content).sum(2).reshape(features.shape)
        x = nn.relu(nn.RMSNorm(epsilon=1e-6, name="after_norm1")(x))
        x = Dense(self.width, use_bias=False, name="after_dense")(x)
        x = nn.RMSNorm(epsilon=1e-6, name="after_norm2")(x)
        return nn.relu(features + x) * valid[..., None]


class TransitionDown(nn.Module):
    width: int
    limit: int

    @nn.compact
    def __call__(self, xyz, features, valid):
        selected, mask = fps(xyz, valid, self.limit)
        centers = gather(xyz, selected)
        ids = knn(xyz, centers, valid, 16)
        relative = gather(xyz, ids) - centers[:, :, None]
        x = Dense(self.width, use_bias=False)(jnp.concatenate((relative, gather(features, ids)), axis=-1))
        x = nn.relu(nn.RMSNorm(epsilon=1e-6)(x))
        x = jnp.where(gather(valid, ids)[..., None], x, -jnp.finfo(x.dtype).max).max(2)
        return centers, x * mask[..., None], mask


class MapEncoder(nn.Module):
    @nn.compact
    def __call__(self, obs, bank):
        ids = obs["feature_ids"].astype(jnp.int32)
        batch_shape, count = ids.shape[:-1], ids.shape[-1]
        ids = ids.reshape(-1, count)
        xyz = obs["xyz"].reshape(-1, count, 3)
        valid = ids >= 0
        # Project the immutable bank before gathering, avoiding B*N*1024 storage.
        projected = Dense(64, name="input")(jax.lax.stop_gradient(bank))
        x = projected[jnp.maximum(ids, 0)]
        for i, (width, limit) in enumerate(((64, 256), (128, 64))):
            xyz, x, valid = TransitionDown(width, limit, name=f"down_{i}")(xyz, x, valid)
            x = PointBlock(width, name=f"block_{i}")(xyz, x, valid)
        x = PointBlock(128, neighbors=64, name="global_block")(xyz, x, valid)
        scores = Dense(1, name="score")(x).squeeze(-1)
        scores = jnp.where(valid, scores, -jnp.finfo(scores.dtype).max)
        pooled = (x * jax.nn.softmax(scores, axis=1)[..., None]).sum(1)
        return Dense(256, name="output")(pooled).reshape(*batch_shape, 256)
