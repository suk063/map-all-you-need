"""Compact relative-only Point Transformer, adapted from SERF-VLA (MIT).

Reference: ExistentialRobotics/SERF-VLA, commit ea27b7aa753cf7da6def975846ccb5d3180e46f7,
src/serf_b1k/models/point_transformer_local.py. See THIRD_PARTY_NOTICES.md.
Coordinates are used only for sampling, neighbours and relative position encoding.
"""

import torch
from torch import nn


def gather(x, ids):
    batch = torch.arange(x.shape[0], device=x.device).view(-1, *([1] * (ids.ndim - 1)))
    return x[batch, ids]


def sampling_coordinates(xyz, origin):
    # A 0.1 mm relative grid stabilizes equal-distance ties on planar surfaces after
    # float32 translations. Attention still uses the original metre-valued offsets.
    return ((xyz - origin) * 10000).round()


@torch.no_grad()
def fps(xyz, valid, limit):
    xyz = sampling_coordinates(xyz, xyz[:, :1])
    count = valid.sum(1)
    size = min(limit, int(count.max()))
    ids = torch.empty(xyz.shape[0], size, device=xyz.device, dtype=torch.long)
    distance = torch.full(valid.shape, float("inf"), device=xyz.device)
    farthest = valid.long().argmax(1)
    for i in range(size):
        ids[:, i] = farthest
        center = gather(xyz, farthest[:, None])
        distance = torch.minimum(distance, ((xyz - center) ** 2).sum(-1))
        farthest = distance.masked_fill(~valid, -1).argmax(1)
    return ids, torch.arange(size, device=xyz.device)[None] < count[:, None]


def knn(xyz, centers, valid, k):
    origin = xyz[:, :1]
    xyz, centers = sampling_coordinates(xyz, origin), sampling_coordinates(centers, origin)
    k = min(k, xyz.shape[1])
    indices = []
    # Direct differences avoid loss of translation invariance from x²+y²-2xy.
    for block in centers.split(32, dim=1):
        distance = ((block[:, :, None] - xyz[:, None]) ** 2).sum(-1)
        indices.append(distance.masked_fill(~valid[:, None], float("inf")).topk(k, largest=False).indices)
    return torch.cat(indices, dim=1)


class PointBlock(nn.Module):
    def __init__(self, width, neighbors=16):
        super().__init__()
        self.neighbors, self.share = neighbors, 8
        self.before = nn.Sequential(nn.Linear(width, width, bias=False), nn.RMSNorm(width, eps=1e-6), nn.ReLU())
        self.qkv = nn.Linear(width, 3 * width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.RMSNorm(width, eps=1e-6), nn.ReLU(), nn.Linear(width, width))
        self.weight = nn.Sequential(nn.RMSNorm(width, eps=1e-6), nn.ReLU(), nn.Linear(width, width // 8),
                                    nn.RMSNorm(width // 8, eps=1e-6), nn.ReLU(), nn.Linear(width // 8, width // 8))
        self.after = nn.Sequential(nn.RMSNorm(width, eps=1e-6), nn.ReLU(), nn.Linear(width, width, bias=False), nn.RMSNorm(width, eps=1e-6))

    def forward(self, xyz, features, valid):
        q, k, v = self.qkv(self.before(features)).chunk(3, dim=-1)
        ids = knn(xyz, xyz, valid, self.neighbors)
        delta = self.position(gather(xyz, ids) - xyz[:, :, None])
        scores = self.weight(gather(k, ids) - q[:, :, None] + delta)
        scores = scores.masked_fill(~gather(valid, ids)[..., None], -torch.finfo(scores.dtype).max)
        content = (gather(v, ids) + delta).unflatten(-1, (self.share, -1))
        x = (scores.softmax(2)[..., None, :] * content).sum(2).flatten(-2)
        return (features + self.after(x)).relu() * valid[..., None]


class TransitionDown(nn.Module):
    def __init__(self, input_dim, width, limit):
        super().__init__()
        self.limit = limit
        self.project = nn.Sequential(nn.Linear(input_dim + 3, width, bias=False), nn.RMSNorm(width, eps=1e-6), nn.ReLU())

    def forward(self, xyz, features, valid):
        selected, mask = fps(xyz, valid, self.limit)
        centers = gather(xyz, selected)
        ids = knn(xyz, centers, valid, 16)
        relative = gather(xyz, ids) - centers[:, :, None]
        x = self.project(torch.cat((relative, gather(features, ids)), dim=-1))
        x = x.masked_fill(~gather(valid, ids)[..., None], -torch.finfo(x.dtype).max).max(2).values
        return centers, x * mask[..., None], mask


class MapEncoder(nn.Module):
    output_dim = 256

    def __init__(self, bank):
        super().__init__()
        self.bank = bank  # Frozen feature storage, not a learned embedding table.
        self.input = nn.Linear(1024, 64)
        self.down = nn.ModuleList([TransitionDown(64, 64, 256), TransitionDown(64, 128, 64)])
        self.blocks = nn.ModuleList([PointBlock(64), PointBlock(128)])
        self.global_block = PointBlock(128, neighbors=64)
        self.score = nn.Linear(128, 1)
        self.output = nn.Linear(128, 256)

    def forward(self, obs):
        xyz, ids = obs["xyz"].float(), obs["feature_ids"]
        valid = ids >= 0
        if not valid.any(1).all():
            raise ValueError("Map observations must contain at least one observed point per environment")
        features = self.bank.lookup(ids).float()
        x = self.input(features)
        for down, block in zip(self.down, self.blocks):
            xyz, x, valid = down(xyz, x, valid)
            x = block(xyz, x, valid)
        x = self.global_block(xyz, x, valid)
        weights = self.score(x).squeeze(-1).masked_fill(~valid, -torch.finfo(x.dtype).max).softmax(1)
        # Exactly one global 256D token per map, returned as the shared encoder vector.
        return self.output((x * weights[..., None]).sum(1))
