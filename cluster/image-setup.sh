#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git ca-certificates libegl1 libgl1 libglib2.0-0 libglvnd0 libopengl0
rm -rf /var/lib/apt/lists/*
mkdir -p /usr/share/glvnd/egl_vendor.d
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
python -m pip install --no-cache-dir -r requirements.txt pytest==9.1.1
git init /opt/dinov3
git -C /opt/dinov3 remote add origin https://github.com/facebookresearch/dinov3.git
git -C /opt/dinov3 fetch --depth 1 origin 6876159a11b4df116f30f667f8c9888617df0751
git -C /opt/dinov3 checkout --detach FETCH_HEAD
python -c 'from mujoco_playground._src.mjx_env import ensure_menagerie_exists; ensure_menagerie_exists()'
