"""Lazy, transition-aligned access to native ManiSkill HDF5 + JSON demos."""

import bisect
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.common.envs import OBS_MODES, TASKS, shared_control_mode
from benchmark.common.policy import at_path, prepare_observation


def read_metadata(path, obs_mode, requested_control_mode=None):
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("Both the trajectory .h5 and its matching .json are required")
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("episodes") or "env_info" not in metadata:
        raise ValueError("Demo metadata needs env_info and nonempty episodes")
    info = metadata["env_info"]
    if info.get("env_id") not in TASKS or not isinstance(info.get("env_kwargs"), dict):
        raise ValueError("Demo env_info must describe a supported dense-reward task")
    kwargs = copy.deepcopy(info["env_kwargs"])
    recorded_mode = kwargs.get("obs_mode")
    if recorded_mode == "rgb+depth":
        recorded_mode = "rgbd"
    if obs_mode != "map" and recorded_mode not in OBS_MODES:
        raise ValueError("Demo has no supported observations; replay it with --obs-mode state, rgb, or rgbd --save-traj")
    native_mode = {"dino": "rgb", "map": "none"}.get(obs_mode, obs_mode)
    if obs_mode != "map" and recorded_mode != native_mode:
        raise ValueError(f"Demo mode is {recorded_mode}, requested {obs_mode}; replay the demo in the requested mode")
    control_mode = kwargs.get("control_mode")
    if control_mode is None:
        control_mode = metadata["episodes"][0].get("control_mode")
    if control_mode is None:
        raise ValueError("Demo metadata is missing control_mode")
    control_mode = shared_control_mode(control_mode)
    if requested_control_mode is not None and requested_control_mode != control_mode:
        raise ValueError(
            f"Demo controller is {control_mode}, requested {requested_control_mode}. "
            f"Use --control-mode {control_mode} to keep these actions, or replay the demo "
            f"with --target-control-mode {requested_control_mode} before training."
        )
    for episode in metadata["episodes"]:
        recorded_control = episode.get("control_mode", control_mode)
        if shared_control_mode(recorded_control) != control_mode:
            raise ValueError("All demonstration episodes must use the same controller")
    # ManiSkill's replay tool uses CPU for legacy metadata without a backend.
    backend = kwargs.get("sim_backend", "physx_cpu")
    if backend == "auto":
        backend = "physx_cuda" if kwargs.get("num_envs", 1) > 1 else "physx_cpu"
    backend = {"cpu": "physx_cpu", "cuda": "physx_cuda", "gpu": "physx_cuda",
               "physx_gpu": "physx_cuda"}.get(backend, backend)
    if backend not in ("physx_cpu", "physx_cuda"):
        raise ValueError(f"Unsupported demo simulation backend: {backend}")
    kwargs.pop("num_envs", None)
    kwargs.update(obs_mode=native_mode, control_mode=control_mode,
                  sim_backend=backend, reward_mode="dense")
    return metadata, {"env_id": info["env_id"], "env_kwargs": kwargs}


def observation_paths(spec):
    paths = list(spec["state_paths"])
    modalities = ("rgb", "depth") if spec["mode"] == "rgbd" else ("rgb",)
    for camera in spec["cameras"]:
        paths.extend([["sensor_data", camera, m] for m in modalities])
    return paths


def read_observation(group, index, spec):
    if spec["mode"] == "state":
        return group[index][None]
    obs = {}
    for path in observation_paths(spec):
        target = obs
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = np.asarray(at_path(group, path)[index])[None]
    return obs


class DemoDataset(Dataset):
    def __init__(self, path, metadata, obs_spec, actions, num_demos=None):
        self.path, self.obs_spec, self.actions = str(path), obs_spec, actions
        self.trajectory_ids, self.ends = [], []
        self._file = None
        if num_demos is not None and num_demos < 1:
            raise ValueError("num_demos must be positive")
        episodes = metadata["episodes"][:num_demos]
        if len({e["episode_id"] for e in episodes}) != len(episodes):
            raise ValueError("Duplicate episode_id in demo metadata")
        with h5py.File(path, "r") as data:
            for episode in episodes:
                name = f"traj_{episode['episode_id']}"
                if name not in data or "obs" not in data[name] or "actions" not in data[name]:
                    raise ValueError(f"{name}: missing obs/actions; replay with --save-traj and the desired --obs-mode")
                trajectory = data[name]
                action_data = trajectory["actions"]
                if isinstance(action_data, h5py.Group):
                    expected = [p["key"] for p in actions]
                    if None in expected or set(action_data) != set(expected):
                        raise ValueError(f"{name}: action keys do not match the environment")
                    arrays = [(action_data[p["key"]], p["size"]) for p in actions]
                else:
                    arrays = [(action_data, sum(p["size"] for p in actions))]
                length = arrays[0][0].shape[0]
                if length < 1 or any(a.shape != (length, size) for a, size in arrays):
                    raise ValueError(f"{name}: invalid action shape or empty trajectory")
                try:
                    for obs_path in observation_paths(obs_spec):
                        leaf = at_path(trajectory["obs"], obs_path)
                        if not isinstance(leaf, h5py.Dataset) or leaf.shape[0] != length + 1:
                            raise ValueError(f"{name}: observations must contain T+1 samples for T actions")
                    prepare_observation(read_observation(trajectory["obs"], 0, obs_spec), obs_spec, "cpu")
                except (KeyError, IndexError, RuntimeError) as exc:
                    raise ValueError(
                        f"{name}: incompatible {obs_spec['mode']} observations or missing selected "
                        f"cameras {obs_spec['cameras']}; replay the demo with these cameras"
                    ) from exc
                self.trajectory_ids.append(name)
                self.ends.append((self.ends[-1] if self.ends else 0) + length)
        if not self.ends:
            raise ValueError("No demonstration transitions")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        trajectory_index = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[trajectory_index - 1] if trajectory_index else 0)
        trajectory = self._file[self.trajectory_ids[trajectory_index]]
        obs = prepare_observation(read_observation(trajectory["obs"], offset, self.obs_spec), self.obs_spec, "cpu")
        data = trajectory["actions"]
        action = (np.concatenate([data[p["key"]][offset] for p in self.actions])
                  if isinstance(data, h5py.Group) else data[offset])
        return {k: v.squeeze(0) for k, v in obs.items()}, torch.as_tensor(action, dtype=torch.float32)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
