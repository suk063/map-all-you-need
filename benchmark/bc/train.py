"""Single-step MSE behavior cloning from replayed ManiSkill demonstrations."""

import argparse
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmark.bc.data import DemoDataset, read_metadata
from benchmark.common.envs import (
    DEFAULT_CONTROL_MODE,
    OBS_MODES,
    action_spec,
    make_env,
    resolved_config,
)
from benchmark.common.policy import (
    Policy,
    add_policy_arguments,
    observation_spec,
    policy_options,
    save_policy,
)
from benchmark.common.utils import log_row, run_directory, seed_everything, write_json


def train(args):
    if min(args.epochs, args.batch_size if args.batch_size is not None else 1, args.save_every) < 1:
        raise ValueError("Training counts must be positive")
    seed_everything(args.seed)
    setup_start = time.monotonic()
    options = policy_options(args)
    if args.batch_size is None:
        args.batch_size = 16 if args.obs_mode == "map" else 256
    metadata, config = read_metadata(args.demo_path, args.obs_mode, args.control_mode)
    if args.obs_mode == "map":
        config["map"] = options["map_config"]
    env = make_env(config, evaluation=True)
    try:
        obs, _ = env.reset(seed=args.seed)
        policy = Policy(observation_spec(obs, args.obs_mode, args.view, **options), action_spec(env),
                        resolved_config(config, env), env.map_bank).to(args.device)
        collate = None
        if args.obs_mode == "map":
            from benchmark.bc.map_data import MapDemoDataset, collate_maps
            dataset = MapDemoDataset(args.demo_path, metadata, env, policy.action_spec, args.num_demos)
            collate = collate_maps
        else:
            dataset = DemoDataset(args.demo_path, metadata, policy.obs_spec, policy.action_spec, args.num_demos)
    finally:
        env.close()
    try:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
        optimizer = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)
        output = run_directory(args.output, "bc", config["env_id"], args.obs_mode, args.seed)
        settings = {**vars(args), "algorithm": "bc", "env_config": policy.env_config,
                    "num_demos": len(dataset.trajectory_ids), "transitions": len(dataset), "obs_spec": policy.obs_spec,
                    "setup_seconds": time.monotonic() - setup_start,
                    "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad)}
        write_json(output / "config.json", settings)
        if policy.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(policy.device)
        start, steps = time.monotonic(), 0
        for epoch in range(1, args.epochs + 1):
            loss_sum, count = 0.0, 0
            for observations, actions in loader:
                observations = {k: v.to(args.device) for k, v in observations.items()}
                actions = actions.to(args.device)
                # Official RL demos can store Gaussian samples BEFORE controller clipping.
                if not torch.isfinite(actions).all():
                    raise ValueError("Demo actions are non-finite")
                loss = F.mse_loss(policy(observations), actions)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite BC loss; check demonstration observations")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(actions)
                count += len(actions)
                steps += 1
            seconds = time.monotonic() - start
            log_row(output / "train.csv", {"epoch": epoch, "steps": steps, "seconds": seconds,
                    "samples_per_second": epoch * len(dataset) / seconds,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(policy.device) / 2**20 if policy.device.type == "cuda" else None,
                    "mse": loss_sum / count})
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_policy(output / "policy.pt", policy, {**settings, "epoch": epoch, "steps": steps})
        print(f"Policy saved: {output / 'policy.pt'}")
        return output / "policy.pt"
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--obs-mode", choices=OBS_MODES, default="state")
    parser.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE,
                        help="Must match the demonstration controller (default: %(default)s)")
    add_policy_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, help="Default: 16 for map, 256 otherwise")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-demos", type=int)
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint interval in epochs")
    parser.add_argument("--output", help="New run directory (must not already exist)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
