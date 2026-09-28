"""Shared observation processing, actor, and self-contained policy checkpoints."""

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def leaf_paths(tree, prefix=()):
    if isinstance(tree, Mapping):
        for key in sorted(tree):
            yield from leaf_paths(tree[key], (*prefix, key))
    else:
        yield list(prefix)


def at_path(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def observation_spec(obs, mode):
    paths = [[]] if mode == "state" else [
        path for root in ("agent", "extra")
        for path in leaf_paths(obs[root], (root,))
    ]
    shapes = [list(at_path(obs, path).shape[1:]) for path in paths]
    return {"mode": mode, "state_paths": paths, "state_shapes": shapes,
            "state_dim": sum(int(np.prod(s)) for s in shapes),
            "cameras": [] if mode == "state" else sorted(obs["sensor_data"]),
            "image_size": 128}


def prepare_observation(obs, spec, device):
    """Accept native, batched ManiSkill observations; keep RGB buffers as uint8."""
    states = []
    for path, shape in zip(spec["state_paths"], spec["state_shapes"]):
        value = torch.as_tensor(at_path(obs, path), device=device)
        if list(value.shape[1:]) != shape:
            raise ValueError(f"Observation shape mismatch at {path}: {value.shape}")
        states.append(value.float().reshape(value.shape[0], -1))
    result = {"state": torch.cat(states, dim=-1)}
    if spec["mode"] == "state":
        return result
    for modality in ("rgb", "depth") if spec["mode"] == "rgbd" else ("rgb",):
        images = []
        for camera in spec["cameras"]:
            x = torch.as_tensor(obs["sensor_data"][camera][modality], device=device)
            channels = 3 if modality == "rgb" else 1
            if x.ndim != 4 or x.shape[-1] != channels:
                raise ValueError(f"Expected batched HWC {modality} with {channels} channels: {camera}")
            x = x.permute(0, 3, 1, 2).float()
            if modality == "rgb":
                x = F.interpolate(x, (spec["image_size"],) * 2, mode="bilinear", align_corners=False)
                x = x.round().clamp(0, 255).to(torch.uint8)
            else:
                x = F.interpolate(x, (spec["image_size"],) * 2, mode="nearest") / 1000.0
            images.append(x)
        result[modality] = torch.cat(images, dim=1)
    return result


class Encoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(spec["state_dim"], 256), nn.Tanh(),
                                   nn.Linear(256, 256), nn.Tanh())
        self.image = None
        self.output_dim = 256
        if spec["mode"] != "state":
            channels = len(spec["cameras"]) * (4 if spec["mode"] == "rgbd" else 3)
            self.image = nn.Sequential(
                nn.Conv2d(channels, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
                nn.Linear(64 * 12 * 12, 256), nn.ReLU(),
            )
            self.output_dim += 256

    def forward(self, obs):
        state = self.state(obs["state"])
        if self.image is None:
            return state
        image = obs["rgb"].float() / 255.0
        if "depth" in obs:
            image = torch.cat((image, obs["depth"].float()), dim=1)
        return torch.cat((state, self.image(image)), dim=-1)


class Policy(nn.Module):
    def __init__(self, obs_spec, actions, config):
        super().__init__()
        self.obs_spec, self.action_spec, self.env_config = obs_spec, actions, config
        self.encoder = Encoder(obs_spec)
        self.mean = nn.Linear(self.encoder.output_dim, sum(p["size"] for p in actions))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((self.mean.out_features,), -0.5))
        self.register_buffer("action_low", torch.tensor([x for p in actions for x in p["low"]]))
        self.register_buffer("action_high", torch.tensor([x for p in actions for x in p["high"]]))

    @property
    def device(self):
        return self.action_low.device

    def prepare(self, obs):
        return prepare_observation(obs, self.obs_spec, self.device)

    def forward(self, prepared_obs):
        return self.mean(self.encoder(prepared_obs))

    def clip(self, action):
        return action.clamp(self.action_low, self.action_high)

    @torch.inference_mode()
    def act(self, obs):
        """Deterministic, bounded action from a native batched observation."""
        return self.clip(self(self.prepare(obs)))


def save_policy(path, policy, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1, "obs_spec": policy.obs_spec,
        "action_spec": policy.action_spec, "env_config": policy.env_config,
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "training": training,
        "versions": {name: version(name) for name in ("mani_skill", "sapien", "torch", "gymnasium", "numpy")},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    policy = Policy(checkpoint["obs_spec"], checkpoint["action_spec"], checkpoint["env_config"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return policy.to(device).eval()
