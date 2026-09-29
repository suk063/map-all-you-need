"""Evaluate either training method: python -m benchmark.eval --checkpoint policy.pt."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from benchmark.common.envs import action_spec, make_env
from benchmark.common.policy import load_policy, observation_spec
from benchmark.common.utils import seed_everything, write_json


@torch.inference_mode()
def evaluate(policy, episodes=100, seed=10000, num_envs=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    cpu_sim = policy.env_config["env_kwargs"]["sim_backend"] == "physx_cpu"
    num_envs = min(episodes, num_envs or (1 if cpu_sim else 8))
    mode = policy.obs_spec["mode"]
    env = make_env(policy.env_config, num_envs, evaluation=True, map_bank=policy.map_bank)
    rows = []
    try:
        obs, _ = env.reset(seed=seed)
        if observation_spec(obs, mode, policy.obs_spec["view"], policy.obs_spec["state_input"],
                            policy.obs_spec.get("dino"), policy.obs_spec.get("map")) != policy.obs_spec:
            raise ValueError("Evaluation observation layout differs from the checkpoint")
        if action_spec(env) != policy.action_spec:
            raise ValueError("Evaluation action layout differs from the checkpoint")
        while len(rows) < episodes:
            obs, _, _, truncated, info = env.step(policy.act(obs).to(env.device))
            if truncated.any():
                metrics = info["final_info"]["episode"]
                for i in torch.where(info["_final_info"])[0].tolist():
                    if len(rows) == episodes:
                        break
                    rows.append({"episode": len(rows), **{
                        k: float(metrics[k][i].item())
                        for k in ("success_once", "success_at_end", "return", "episode_len")
                    }})
    finally:
        env.close()
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "episode"}
    return {"episodes": episodes, "seed": seed, "num_envs": num_envs,
            "view": policy.obs_spec["view"], "cameras": policy.obs_spec["cameras"],
            "obs_spec": policy.obs_spec, "env_config": policy.env_config, "metrics": summary}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Output prefix; defaults to <checkpoint directory>/eval-seed<seed>")
    args = parser.parse_args()
    seed_everything(args.seed)
    policy = load_policy(args.checkpoint, args.device)
    summary, rows = evaluate(policy, args.episodes, args.seed, args.num_envs)
    summary.update(checkpoint=str(Path(args.checkpoint).resolve()),
                   training=policy.metadata["training"], versions=policy.metadata["versions"],
                   device=args.device)
    prefix = Path(args.output) if args.output else Path(args.checkpoint).parent / f"eval-seed{args.seed}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(prefix) + ".json", summary)
    with open(str(prefix) + ".csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary["metrics"])
    print(f"Saved {prefix}.json and {prefix}.csv")


if __name__ == "__main__":
    main()
