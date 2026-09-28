"""Subprocess workers for GPU integration tests (fixtures are not expert demos)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from benchmark.common.envs import (
    OBS_MODES,
    action_spec,
    env_config,
    make_env,
    resolved_config,
)
from benchmark.common.policy import Policy, load_policy, observation_spec, save_policy
from benchmark.common.utils import seed_everything, write_json


def environment_smoke(env_id):
    for mode in OBS_MODES:
        config = env_config(env_id, mode)
        env = make_env(config, 2, evaluation=True)
        try:
            obs, _ = env.reset(seed=42)
            policy = Policy(observation_spec(obs, mode), action_spec(env), resolved_config(config, env)).cuda()
            assert env.unwrapped.reward_mode == "dense"
            for _ in range(2):
                prepared = policy.prepare(obs)
                assert all(torch.isfinite(v).all() for v in prepared.values())
                action = policy.act(obs)
                assert action.shape == (2, sum(p["size"] for p in policy.action_spec))
                obs, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all()
            if env_id.startswith("TwoRobot"):
                assert len(policy.action_spec) == 2
                assert len(policy.obs_spec["cameras"]) >= 2 or mode == "state"
            print(f"PASS {env_id} {mode}", flush=True)
        finally:
            env.close()
        del env, policy, obs


def snapshot(tree):
    if isinstance(tree, dict):
        return {k: snapshot(v) for k, v in tree.items()}
    return tree.detach().cpu().numpy().copy()


def write_observations(group, observations):
    for key in observations[0]:
        values = [o[key] for o in observations]
        if isinstance(values[0], dict):
            write_observations(group.create_group(key), values)
        else:
            group.create_dataset(key, data=np.concatenate(values))


def create_test_demo(path, mode, env_id="PickCube-v1"):
    config = env_config(env_id, mode)
    config["env_kwargs"]["sim_backend"] = "physx_cpu"
    env = make_env(config, evaluation=True)
    try:
        metadata = {"env_info": resolved_config(config, env), "episodes": [],
                    "source_type": "test", "source_desc": "Zero-action smoke fixture; not expert demonstration data"}
        with h5py.File(path, "w") as data:
            for episode_id in range(2):
                obs, _ = env.reset(seed=episode_id)
                observations = [snapshot(obs)]
                actions = []
                for _ in range(6):
                    action = torch.zeros((1, sum(p["size"] for p in action_spec(env))))
                    obs, _, _, _, _ = env.step(action)
                    actions.append(action.numpy())
                    observations.append(snapshot(obs))
                group = data.create_group(f"traj_{episode_id}")
                if mode == "state":
                    group.create_dataset("obs", data=np.concatenate(observations))
                else:
                    write_observations(group.create_group("obs"), observations)
                if env_id.startswith("TwoRobot"):
                    offset = 0
                    parts = group.create_group("actions")
                    for p in action_spec(env):
                        parts.create_dataset(p["key"], data=np.concatenate(actions)[:, offset:offset+p["size"]])
                        offset += p["size"]
                else:
                    group.create_dataset("actions", data=np.concatenate(actions))
                metadata["episodes"].append({"episode_id": episode_id,
                    "control_mode": config["env_kwargs"]["control_mode"], "elapsed_steps": 6})
        write_json(path.with_suffix(".json"), metadata)
    finally:
        env.close()


def command(*args):
    subprocess.run([sys.executable, *args], check=True, timeout=180)


def training_smoke(mode, output):
    output.mkdir(parents=True, exist_ok=True)
    command("-m", "benchmark.rl.train", "--obs-mode", mode, "--num-envs", "2", "--num-steps", "50",
            "--total-timesteps", "104", "--update-epochs", "1", "--batch-size", "32",
            "--output", str(output / "rl"))
    demo = output / "trajectory.h5"
    create_test_demo(demo, mode)
    command("-m", "benchmark.bc.train", "--obs-mode", mode, "--demo-path", str(demo),
            "--epochs", "1", "--batch-size", "4", "--output", str(output / "bc"))
    for method in ("rl", "bc"):
        checkpoint = output / method / "policy.pt"
        command("-m", "benchmark.eval", "--checkpoint", str(checkpoint), "--episodes", "3")
        summary = json.loads((checkpoint.parent / "eval-seed10000.json").read_text())
        assert summary["episodes"] == 3
        assert summary["metrics"]["episode_len"] == 50
        assert all(np.isfinite(v) for v in summary["metrics"].values())
        # Check actual native observations after re-saving and loading the actor.
        policy = load_policy(checkpoint)
        config = policy.env_config.copy()
        config["env_kwargs"] = {**config["env_kwargs"], "sim_backend": "physx_cpu"}
        env = make_env(config, evaluation=True)
        try:
            obs, _ = env.reset(seed=10)
            copy_path = output / f"{method}-copy.pt"
            save_policy(copy_path, policy, policy.metadata["training"])
            torch.testing.assert_close(policy.act(obs), load_policy(copy_path).act(obs), rtol=0, atol=0)
        finally:
            env.close()
    print(f"PASS train/save/load/eval RL+BC {mode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("env", "train"))
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=OBS_MODES, default="state")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed_everything(0)
    if args.kind == "env":
        environment_smoke(args.env_id)
    else:
        training_smoke(args.obs_mode, args.output)
