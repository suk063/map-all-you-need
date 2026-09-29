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
from benchmark.common.evaluation import DEFINITIONS, VERSION, EvaluationWrapper
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
    base = EvaluationWrapper(base, config["env_id"])
    env = wrapper.wrap_for_brax_training(base, params["episode_length"], params["action_repeat"])

    @jax.jit
    def rollout(keys):
        state = env.reset(keys)
        initial = (state, jnp.ones(num_envs, dtype=bool), state.info['_eval_stats'])

        def step(carry):
            state, active, stats = carry
            action = policy.inference(state.obs, jax.random.PRNGKey(0))[0]
            state = env.step(state, action)
            stats = {key: jnp.where(active, value, stats[key])
                     for key, value in state.info['_eval_stats'].items()}
            return state, active & ~state.done.astype(bool), stats

        return jax.lax.while_loop(lambda carry: carry[1].any(), step, initial)[2]

    rng = jax.random.PRNGKey(seed)
    rows = []
    for _ in range(math.ceil(episodes / num_envs)):
        rng, key = jax.random.split(rng)
        stats = jax.device_get(rollout(jax.random.split(key, num_envs)))
        for i in range(min(num_envs, episodes - len(rows))):
            values = {key: float(value[i]) for key, value in stats.items()}
            steps = max(values['native_steps'], 1)
            row = {'episode': len(rows), 'return': values['return'],
                   'episode_len': values['native_steps'],
                   'duration_seconds': values['native_steps'] * base.native.dt,
                   'success_once': values['success_once'], 'success_final': values['success_final'],
                   'success_count': values['success_count'],
                   'success_fraction': values['success_steps'] / steps,
                   'first_success_seconds': values['first_success_seconds'] if values['success_once'] else None,
                   'terminated': values['terminated'], 'time_limit': 1 - values['terminated'],
                   'termination_without_success': values['terminated'] * (1 - values['success_once']),
                   'nonfinite': values['nonfinite'], 'out_of_bounds_once': values['out_of_bounds_once'],
                   'action_l2_mean': values['action_l2_sum'] / steps,
                   'action_delta_l2_mean': values['action_delta_l2_sum'] / steps,
                   'action_saturation_fraction': values['action_saturation_sum'] / steps}
            for key, value in values.items():
                if key.startswith('native_') and '/' in key:
                    row[key.removeprefix('native_')] = value
                    if key.startswith('native_sum/'):
                        row['mean/' + key[len('native_sum/'):]] = value / steps
                elif '/' in key:
                    row['task/' + key] = value
                    if key.startswith('sum/'):
                        row['task/mean/' + key[4:]] = value / steps
            if config['env_id'].endswith('RotateZAxis'):
                row['net_rotation_rad'] = values['rotation_z_rad']
                row['net_rotations'] = values['rotation_z_rad'] / (2 * math.pi)
            rows.append({k: (v if v is None or math.isfinite(v) else None) for k, v in row.items()})
    metrics, distributions = {}, {}
    for key in rows[0]:
        if key == 'episode':
            continue
        values = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        metrics[key] = float(values.mean()) if len(values) else None
        distributions[key] = {'count': len(values), 'std': float(values.std()) if len(values) else None,
                              'median': float(np.median(values)) if len(values) else None,
                              'p10': float(np.quantile(values, .1)) if len(values) else None,
                              'p90': float(np.quantile(values, .9)) if len(values) else None}
    rate, n, z = metrics['success_once'], len(rows), 1.96
    center = (rate + z*z/(2*n)) / (1+z*z/n)
    radius = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1+z*z/n)
    summary = {'episodes': episodes, 'seed': seed, 'num_envs': num_envs,
               'env_config': config, 'checkpoint': str(policy.checkpoint),
               'evaluation_version': VERSION, 'success_definition': DEFINITIONS[config['env_id']],
               'success_once_ci95': [max(0., center-radius), min(1., center+radius)],
               'metrics': metrics, 'distributions': distributions}
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
