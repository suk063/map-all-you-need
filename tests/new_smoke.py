"""GPU checks using real local DINO weights; eight views keep integration tests short."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmark.bc.data import read_metadata
from benchmark.bc.map_data import MapDemoDataset, read_state
from benchmark.common.envs import action_spec, env_config, make_env, resolved_config
from benchmark.common.policy import (
    Policy,
    add_policy_arguments,
    load_policy,
    observation_spec,
    policy_options,
    save_policy,
)
from benchmark.common.utils import seed_everything, write_json
from tests.smoke import create_test_demo


def map_demo_alignment(args):
    import h5py
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "trajectory.h5"
    create_test_demo(path, "state", moving=True)
    metadata, config = read_metadata(path, "map")
    config["map"] = policy_options(args)["map_config"]
    env = make_env(config, evaluation=True)
    try:
        env.reset(seed=17)
        dataset = MapDemoDataset(path, metadata, env, action_spec(env))
        try:
            with h5py.File(path) as f:
                for episode in range(2):
                    env.reset(seed=episode)
                    for t in (0, 1, 5):
                        group = f[f"traj_{episode}"]
                        env.unwrapped.set_state_dict(read_state(group["env_states"], t))
                        online = env.map_wrapper.observation()
                        offline, action = dataset[episode * 6 + t]
                        torch.testing.assert_close(offline["xyz"], online["xyz"][0], rtol=0, atol=0)
                        torch.testing.assert_close(env.map_bank.lookup(offline["feature_ids"]), env.map_bank.lookup(online["feature_ids"])[0], rtol=0, atol=0)
                        torch.testing.assert_close(action, torch.from_numpy(group["actions"][t]), rtol=0, atol=0)
            # The same cache must work with a differently ordered feature bank as well.
            from benchmark.common.mapping import FeatureBank
            old_bank = env.map_bank
            env.map_wrapper.bank = env.map_bank = FeatureBank(reversed(old_bank.paths))
            cached = MapDemoDataset(path, metadata, env, action_spec(env))
            try:
                for t in (0, 5, 6, 11):
                    old, new = dataset[t][0], cached[t][0]
                    torch.testing.assert_close(old["xyz"], new["xyz"], rtol=0, atol=0)
                    torch.testing.assert_close(old_bank.lookup(old["feature_ids"]), env.map_bank.lookup(new["feature_ids"]), rtol=0, atol=0)
            finally:
                cached.close()
        finally:
            dataset.close()
    finally:
        env.close()
    print("PASS map BC state/action alignment and online preprocessing", flush=True)


def map_environment(args):
    options = policy_options(args)
    config = env_config(args.env_id, "map")
    config["map"] = options["map_config"]
    config["env_kwargs"]["max_episode_steps"] = 2
    env = make_env(config, 2, evaluation=True)
    start = time.monotonic()
    try:
        env.map_wrapper.env.reset(seed=42)
        before = env.unwrapped.get_state_dict()
        before = {kind: {k: v.clone() for k, v in values.items()} for kind, values in before.items()}
        rng = torch.get_rng_state()
        env.map_wrapper.bind()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        after = env.unwrapped.get_state_dict()
        for kind, values in before.items():
            for key, value in values.items():
                torch.testing.assert_close(after[kind][key], value, rtol=0, atol=0)
        names = {b[0].name for b in env.map_wrapper.bindings}
        assert ("table-workspace" in names) == (args.map_background == "table")
        for actor in env.unwrapped._hidden_objects:
            assert not actor.has_collision_shapes
            assert actor.name in names
        obs, _ = env.reset(seed=42)
        build_seconds = time.monotonic() - start
        policy = Policy(observation_spec(obs, "map", **options), action_spec(env), resolved_config(config, env), env.map_bank).cuda()
        points = (obs["feature_ids"] >= 0).sum(1).tolist()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            action = policy.act(obs)
            old = {k: v.clone() for k, v in obs.items()}
            obs, reward, _, truncated, info = env.step(action)
            assert torch.isfinite(reward).all()
            assert action.shape == (2, sum(p["size"] for p in action_spec(env)))
            if truncated.any():
                final = info["final_observation"]
                assert torch.isfinite(policy.act(final)).all()
                assert not torch.equal(final["xyz"], obs["xyz"])
            assert torch.isfinite(policy.act(old)).all()
        token = policy.encoder(policy.prepare(obs))
        assert token.shape == (2, 256)
        token.square().mean().backward()
        report = {"env_id": args.env_id, "robot": args.map_robot, "background": args.map_background,
                  "points": points, "encoder_parameters": sum(p.numel() for p in policy.encoder.parameters()),
                  "map_setup_seconds": build_seconds, "total_seconds": time.monotonic() - start,
                  "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                  "views": args.map_views, "extra_views": args.map_extra_views}
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report), flush=True)
    finally:
        env.close()


def train_and_evaluate(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    flags = ["--obs-mode", args.obs_mode, "--view", args.view]
    if args.state_input is not None:
        flags += ["--state-input" if args.state_input else "--no-state-input"]
    if args.obs_mode == "map":
        flags += ["--map-robot", args.map_robot, "--map-background", args.map_background,
                  "--map-views", str(args.map_views), "--map-extra-views", str(args.map_extra_views), "--map-cache", args.map_cache]
    def command(*arguments):
        subprocess.run([sys.executable, *arguments], check=True, timeout=900)
    command("-m", "benchmark.rl.train", *flags, "--num-envs", "2", "--total-timesteps", "104",
            "--num-steps", "50", "--update-epochs", "1", "--batch-size", "8", "--output", str(output / "rl"))
    path = output / "trajectory.h5"
    create_test_demo(path, "state" if args.obs_mode == "map" else "rgb" if args.obs_mode == "dino" else args.obs_mode)
    command("-m", "benchmark.bc.train", *flags, "--demo-path", str(path), "--epochs", "1", "--batch-size", "4",
            "--output", str(output / "bc"))
    for method in ("rl", "bc"):
        path = output / method / "policy.pt"
        command("-m", "benchmark.eval", "--checkpoint", str(path), "--episodes", "2", "--num-envs", "1")
        policy = load_policy(path, "cuda")
        bank = policy.encoder.bank if args.obs_mode == "map" else None
        config = {**policy.env_config, "env_kwargs": {**policy.env_config["env_kwargs"], "sim_backend": "physx_cpu"}}
        env = make_env(config, evaluation=True, map_bank=bank)
        try:
            obs, _ = env.reset(seed=31)
            duplicate = output / f"{method}-copy.pt"
            save_policy(duplicate, policy, policy.metadata["training"])
            loaded = load_policy(duplicate, "cuda")
            torch.testing.assert_close(policy.act(obs), loaded.act(obs), rtol=0, atol=0)
            if args.obs_mode == "dino":
                assert policy.prepare(obs)["dino"].shape[1:] == (64, 384)
                assert all(not p.requires_grad for p in policy.encoder.image.backbone.parameters())
        finally:
            env.close()
        summary = json.loads((path.parent / "eval-seed10000.json").read_text())
        assert summary["metrics"]["episode_len"] == 50
    print(f"PASS {args.obs_mode} RL/BC/save/load/eval", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("env", "train", "align"))
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=("map", "dino", "rgb", "rgbd"), default="map")
    parser.add_argument("--output", type=Path)
    add_policy_arguments(parser)
    parser.set_defaults(map_views=8, map_extra_views=0, map_cache=".cache/test-maps")
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(4)
    if args.kind == "env":
        map_environment(args)
    elif args.kind == "align":
        map_demo_alignment(args)
    else:
        train_and_evaluate(args)
