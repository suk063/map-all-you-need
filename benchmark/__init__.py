"""MuJoCo Playground observation benchmarks."""

import os

# Leave room for the batch renderer and the offline DINO feature extractor.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
