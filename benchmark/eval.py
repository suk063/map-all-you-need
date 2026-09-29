"""Evaluate the latest Brax checkpoint in a benchmark run directory."""

import argparse
import csv
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import wrapper

from benchmark.common.envs import make_env
from benchmark.common.policy import load_policy
from benchmark.common.utils import write_json


def evaluate(policy, episodes=100, seed=10000, num_envs=8):
    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    num_envs = min(episodes, num_envs)
    config, params = policy.env_config, policy.metadata["ppo"]
    base = make_env(config, num_envs)
    if base.action_size != policy.metadata["action_size"]:
        raise ValueError("Environment action size differs from the checkpoint")
    if config["obs_mode"] == "map" and base.bank.paths != policy.metadata["map_cache_paths"]:
        raise ValueError("Environment map layout differs from the checkpoint")
    env = wrapper.wrap_for_brax_training(base, params["episode_length"], params["action_repeat"])

    @jax.jit
    def rollout(keys):
        state = env.reset(keys)
        zeros = jnp.zeros(num_envs)
        metric_zeros = {key: zeros for key in state.metrics}
        initial = (state, jnp.ones(num_envs, dtype=bool), zeros, zeros, metric_zeros, metric_zeros)

        def step(carry):
            state, active, returns, lengths, sums, last = carry
            action = policy.inference(state.obs, jax.random.PRNGKey(0))[0]
            state = env.step(state, action)
            returns += active * state.reward
            lengths += active * params["action_repeat"]
            sums = {key: sums[key] + active * value for key, value in state.metrics.items()}
            last = {key: jnp.where(active, value, last[key]) for key, value in state.metrics.items()}
            return state, active & ~state.done.astype(bool), returns, lengths, sums, last

        result = jax.lax.while_loop(lambda carry: carry[1].any(), step, initial)
        return result[2:]

    rng = jax.random.PRNGKey(seed)
    rows = []
    for _ in range(math.ceil(episodes / num_envs)):
        rng, key = jax.random.split(rng)
        returns, lengths, sums, last = jax.device_get(rollout(jax.random.split(key, num_envs)))
        for i in range(min(num_envs, episodes - len(rows))):
            rows.append({"episode": len(rows), "return": float(returns[i]), "episode_len": int(lengths[i]),
                         **{f"sum/{key}": float(value[i]) for key, value in sums.items()},
                         **{f"final/{key}": float(value[i]) for key, value in last.items()}})
    summary = {"episodes": episodes, "seed": seed, "num_envs": num_envs,
               "env_config": config, "checkpoint": str(policy.checkpoint),
               "metrics": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "episode"}}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Run directory containing config.json and checkpoints/")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", help="Output prefix (default: <run>/eval-seed<seed>)")
    args = parser.parse_args()
    policy = load_policy(args.checkpoint)
    summary, rows = evaluate(policy, args.episodes, args.seed, args.num_envs)
    prefix = Path(args.output) if args.output else Path(args.checkpoint) / f"eval-seed{args.seed}"
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
