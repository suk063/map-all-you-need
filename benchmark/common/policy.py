"""Shared observation processing, actors, and policy checkpoints."""

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmark.common.dino import FrozenDINO, dino_config

VIEWS = ("external", "wrist", "all")


def add_policy_arguments(parser):
    import argparse
    parser.add_argument("--view", choices=VIEWS, default="all", help="Cameras for rgb/rgbd/dino")
    parser.add_argument("--state-input", action=argparse.BooleanOptionalAction, default=None,
                        help="Include agent/extra for image policies (default: yes)")
    parser.add_argument("--dino-source", help="Local DINOv3 checkout with hubconf.py")
    parser.add_argument("--dino-weights", help="Local S+/16 weights for dino, L/16 weights for map")
    parser.add_argument("--map-robot", choices=("full", "gripper"), default="full")
    parser.add_argument("--map-background", choices=("table", "none"), default="none",
                        help="Include tabletop points in the map (default: none)")
    parser.add_argument("--map-cache", default=".cache/maps")
    parser.add_argument("--map-views", type=int, default=96)
    parser.add_argument("--map-extra-views", type=int, default=512)


def policy_options(args):
    mode = args.obs_mode
    if (mode == "state" and args.state_input is False) or (mode == "map" and args.state_input is True):
        raise ValueError("state mode requires state input; map mode forbids state input")
    if mode in ("state", "map") and args.view != "all":
        raise ValueError("--view external/wrist requires an image policy")
    options = {"state_input": args.state_input}
    if mode in ("dino", "map"):
        dino = dino_config(mode, args.dino_source, args.dino_weights)
        if mode == "dino":
            options["dino"] = dino
        else:
            if args.map_views < 1 or args.map_extra_views < 0:
                raise ValueError("Map views must be positive and extra views nonnegative")
            options["map_config"] = {"robot": args.map_robot, "background": args.map_background,
                "voxel_size": .015, "views": args.map_views, "extra_views": args.map_extra_views,
                "cache": str(Path(args.map_cache).expanduser().resolve()), "dino": dino}
    return options


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


def observation_spec(obs, mode, view="all", state_input=None, dino=None, map_config=None):
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}; choose from {VIEWS}")
    if mode in ("state", "map") and view != "all":
        raise ValueError("--view external/wrist requires --obs-mode rgb or rgbd or dino")
    if state_input is None:
        state_input = mode != "map"
    if (mode == "state" and not state_input) or (mode == "map" and state_input):
        raise ValueError("state mode requires state input; map mode forbids state input")
    if mode == "map":
        if map_config is None:
            raise ValueError("Map configuration is required")
        return {"mode": mode, "view": view, "state_input": False, "state_paths": [],
                "state_shapes": [], "state_dim": 0, "cameras": [], "image_size": 256, "map": map_config}
    cameras = []
    if mode != "state":
        available = sorted(obs["sensor_data"])
        # ManiSkill's single- and dual-arm wrist sensor UIDs end in these names.
        wrist = [name for name in available if name.endswith(("hand_camera", "wrist_camera"))]
        cameras = available if view == "all" else wrist if view == "wrist" else [
            name for name in available if name not in wrist
        ]
        if not cameras:
            raise ValueError(f"No cameras for view={view!r}. Available cameras: {available}")
    paths = [[]] if mode == "state" else [
        path for root in ("agent", "extra")
        for path in leaf_paths(obs[root], (root,))
    ] if state_input else []
    shapes = [list(at_path(obs, path).shape[1:]) for path in paths]
    spec = {"mode": mode, "state_input": state_input, "state_paths": paths, "state_shapes": shapes,
            "state_dim": sum(int(np.prod(s)) for s in shapes),
            "cameras": cameras, "view": view, "image_size": 128}
    if mode == "dino":
        if dino is None:
            raise ValueError("DINO configuration is required")
        spec["dino"] = dino
    return spec


def prepare_observation(obs, spec, device):
    """Accept native, batched ManiSkill observations; keep RGB buffers as uint8."""
    if spec["mode"] == "map":
        return {"xyz": torch.as_tensor(obs["xyz"], device=device, dtype=torch.float32),
                "feature_ids": torch.as_tensor(obs["feature_ids"], device=device, dtype=torch.long)}
    states = []
    for path, shape in zip(spec["state_paths"], spec["state_shapes"]):
        value = torch.as_tensor(at_path(obs, path), device=device)
        if list(value.shape[1:]) != shape:
            raise ValueError(f"Observation shape mismatch at {path}: {value.shape}")
        states.append(value.float().reshape(value.shape[0], -1))
    result = {"state": torch.cat(states, dim=-1)} if states else {}
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


class DinoImage(nn.Module):
    def __init__(self, spec, pretrained):
        super().__init__()
        self.backbone = FrozenDINO(spec["dino"], pretrained)
        self.patch = nn.Linear(384, 64)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(len(spec["cameras"]) * 64 * 64, 256), nn.ReLU())

    def extract(self, rgb):
        b, channels, h, w = rgb.shape
        return self.backbone(rgb.reshape(b * (channels // 3), 3, h, w)).reshape(b, -1, 384)

    def forward(self, obs):
        features = obs["dino"] if "dino" in obs else self.extract(obs["rgb"])
        return self.head(self.patch(features).relu())


class Encoder(nn.Module):
    def __init__(self, spec, pretrained=True):
        super().__init__()
        self.state = (nn.Sequential(nn.Linear(spec["state_dim"], 256), nn.Tanh(),
                                   nn.Linear(256, 256), nn.Tanh()) if spec["state_input"] else None)
        self.image = None
        self.output_dim = 256 if self.state is not None else 0
        if spec["mode"] == "dino":
            self.image = DinoImage(spec, pretrained)
            self.output_dim += 256
        elif spec["mode"] != "state":
            channels = len(spec["cameras"]) * (4 if spec["mode"] == "rgbd" else 3)
            self.image = nn.Sequential(
                nn.Conv2d(channels, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
                nn.Linear(64 * 12 * 12, 256), nn.ReLU(),
            )
            self.output_dim += 256

    def forward(self, obs):
        state = self.state(obs["state"]) if self.state is not None else None
        if self.image is None:
            return state
        if isinstance(self.image, DinoImage):
            image = self.image(obs)
            return torch.cat((state, image), -1) if state is not None else image
        image = obs["rgb"].float() / 255.0
        if "depth" in obs:
            image = torch.cat((image, obs["depth"].float()), dim=1)
        image = self.image(image)
        return torch.cat((state, image), dim=-1) if state is not None else image


class Policy(nn.Module):
    def __init__(self, obs_spec, actions, config, map_bank=None, pretrained=True):
        super().__init__()
        # Checkpoints created before camera selection always used all cameras.
        self.obs_spec = {"view": "all", "state_input": obs_spec["mode"] != "map", **obs_spec}
        self.action_spec, self.env_config = actions, config
        if obs_spec["mode"] == "map":
            from benchmark.common.mapping import FeatureBank
            from benchmark.common.points import MapEncoder
            self.encoder = MapEncoder(map_bank if map_bank is not None else FeatureBank())
        else:
            self.encoder = Encoder(self.obs_spec, pretrained)
        self.mean = nn.Linear(self.encoder.output_dim, sum(p["size"] for p in actions))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((self.mean.out_features,), -0.5))
        self.register_buffer("action_low", torch.tensor([x for p in actions for x in p["low"]]))
        self.register_buffer("action_high", torch.tensor([x for p in actions for x in p["high"]]))

    @property
    def device(self):
        return self.action_low.device

    @property
    def map_bank(self):
        return getattr(self.encoder, "bank", None)

    def prepare(self, obs):
        result = prepare_observation(obs, self.obs_spec, self.device)
        if self.obs_spec["mode"] == "dino":
            result["dino"] = self.encoder.image.extract(result.pop("rgb"))
        return result

    def forward(self, prepared_obs):
        return self.mean(self.encoder(prepared_obs))

    def clip(self, action):
        return action.clamp(self.action_low, self.action_high)

    @torch.inference_mode()
    def act(self, obs):
        """Deterministic, bounded action from a batched environment observation."""
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
    if policy.obs_spec["mode"] == "map":
        checkpoint["map_features"] = policy.encoder.bank.paths
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    bank = None
    if checkpoint["obs_spec"]["mode"] == "map":
        from benchmark.common.mapping import FeatureBank
        bank = FeatureBank(checkpoint["map_features"])
    policy = Policy(checkpoint["obs_spec"], checkpoint["action_spec"], checkpoint["env_config"], bank, pretrained=False)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return policy.to(device).eval()
