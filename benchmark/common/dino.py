"""Local, frozen DINOv3 ViT-L/16 for map cache generation."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "reachy_task"


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dino_config(source=None, weights=None):
    filename = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    source = Path(source or os.environ.get("DINO_SOURCE", DEFAULT_ROOT / "external/dinov3")).expanduser().resolve()
    weights = Path(weights or os.environ.get("DINO_WEIGHTS", DEFAULT_ROOT / "assets/dinov3" / filename)).expanduser().resolve()
    if not (source / "dinov3/hub/backbones.py").is_file() or not weights.is_file():
        raise ValueError("Local DINOv3 source/weights are missing; set --dino-source and --dino-weights")
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {"model": "dinov3_vitl16", "source": str(source), "weights": str(weights), "sha256": file_hash(weights),
            "source_revision": revision.stdout.strip() if revision.returncode == 0 else file_hash(source / "dinov3/hub/backbones.py")}


class FrozenDINO(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Import only the official backbone; hubconf also imports unrelated task heads.
        sys.path.insert(0, config["source"])
        try:
            from dinov3.hub.backbones import dinov3_vitl16
            self.net = dinov3_vitl16(pretrained=False)
        finally:
            sys.path.pop(0)
        if file_hash(config["weights"]) != config["sha256"]:
            raise ValueError("DINO weights differ from the recorded SHA-256")
        self.net.load_state_dict(torch.load(config["weights"], map_location="cpu", weights_only=True))
        self.net.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb):
        """NCHW uint8 RGB -> normalized patch tokens, with the spatial order intact."""
        x = (rgb.float() / 255 - self.mean) / self.std
        return self.net.forward_features(x)["x_norm_patchtokens"]
