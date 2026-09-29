"""Replay recorded simulator states once; randomly access their cached map snapshots."""

import bisect
import copy
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.common.dino import file_hash
from benchmark.common.mapping import batch_observations


def read_state(group, index):
    return {k: read_state(v, index) if isinstance(v, h5py.Group) else np.asarray(v[index])[None]
            for k, v in group.items()}


def collate_maps(samples):
    observations, actions = zip(*samples)
    return batch_observations([{k: v[None] for k, v in obs.items()} for obs in observations]), torch.stack(actions)


class MapDemoDataset(Dataset):
    def __init__(self, path, metadata, env, actions, num_demos=None):
        if num_demos is not None and num_demos < 1:
            raise ValueError("num_demos must be positive")
        episodes = metadata["episodes"][:num_demos]
        if len({e["episode_id"] for e in episodes}) != len(episodes):
            raise ValueError("Duplicate episode_id in demo metadata")
        self.actions, self._file = actions, None
        self.trajectory_ids = [f"traj_{e['episode_id']}" for e in episodes]
        self.ends = []
        with h5py.File(path) as source:
            for name in self.trajectory_ids:
                if name not in source or "env_states" not in source[name] or "actions" not in source[name]:
                    raise ValueError(f"{name}: map BC requires recorded env_states and actions")
                traj = source[name]
                data = traj["actions"]
                if isinstance(data, h5py.Group):
                    if set(data) != {p["key"] for p in actions}:
                        raise ValueError(f"{name}: action keys differ from the environment")
                    arrays = [(data[p["key"]], p["size"]) for p in actions]
                else:
                    arrays = [(data, sum(p["size"] for p in actions))]
                length = arrays[0][0].shape[0]
                if length < 1 or any(a.shape != (length, size) for a, size in arrays):
                    raise ValueError(f"{name}: invalid action dimensions")
                def check_state(key, value, length=length, name=name):
                    if isinstance(value, h5py.Dataset) and (value.shape[0] != length + 1 or not np.isfinite(value[:]).all()):
                        raise ValueError(f"{name}/{key}: env_states must contain finite T+1 samples")
                traj["env_states"].visititems(check_state)
                if "actors" not in traj["env_states"] or "articulations" not in traj["env_states"]:
                    raise ValueError(f"{name}: env_states lacks actors/articulations")
                self.ends.append((self.ends[-1] if self.ends else 0) + length)
        provenance = {"demo": file_hash(path), "metadata": metadata, "episodes": self.trajectory_ids,
                      "map": env.map_wrapper.config, "format": 1}
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        self.path = str(Path(env.map_wrapper.config["cache"]) / "demos" / f"{key}.h5")
        destination = Path(self.path)
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            with h5py.File(path) as source, h5py.File(temporary, "w") as out:
                for episode, name in zip(episodes, self.trajectory_ids):
                    reset = copy.deepcopy(episode.get("reset_kwargs", {}))
                    if "seed" not in reset:
                        if "episode_seed" not in episode:
                            raise ValueError(f"{name}: episode_seed/reset seed is required for map BC")
                        reset["seed"] = episode["episode_seed"]
                    if isinstance(reset["seed"], list):
                        if len(reset["seed"]) != 1:
                            raise ValueError(f"{name}: expected one reset seed")
                        reset["seed"] = reset["seed"][0]
                    env.reset(**reset)
                    traj = source[name]
                    data = traj["actions"]
                    target = (np.concatenate([data[p["key"]][:] for p in actions], axis=-1)
                              if isinstance(data, h5py.Group) else data[:])
                    if not np.isfinite(target).all():
                        raise ValueError(f"{name}: nonfinite actions")
                    group = out.create_group(name)
                    group.create_dataset("actions", data=target)
                    for t in range(len(target)):
                        env.unwrapped.set_state_dict(read_state(traj["env_states"], t))
                        obs = env.map_wrapper.observation()
                        if t == 0:
                            group.create_dataset("xyz", shape=(len(target), *obs["xyz"].shape[1:]), dtype="f4", compression="gzip")
                            group.create_dataset("feature_ids", data=obs["feature_ids"][0].cpu().numpy())
                        group["xyz"][t] = obs["xyz"][0].cpu().numpy()
                out.attrs["map_features"] = json.dumps(env.map_bank.paths)
                out.attrs["provenance"] = json.dumps(provenance)
            temporary.replace(destination)
        with h5py.File(self.path) as cached:
            paths = json.loads(cached.attrs["map_features"])
        self.remap = torch.cat([env.map_bank.add(p)[1] for p in paths])

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
        ids = torch.from_numpy(trajectory["feature_ids"][:])
        ids = torch.where(ids >= 0, self.remap[ids.clamp_min(0)], -1)
        return {"xyz": torch.from_numpy(trajectory["xyz"][offset]), "feature_ids": ids}, torch.from_numpy(trajectory["actions"][offset]).float()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
