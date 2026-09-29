"""Train official Playground/Brax PPO with state, RGB or a component map."""

import argparse
import json
import math
import time
from importlib.metadata import distribution, version
from pathlib import Path

import jax
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src import wrapper

from benchmark.common.envs import (
    DEFAULT_TASK,
    OBS_MODES,
    TASKS,
    env_config,
    make_env,
    ppo_config,
)
from benchmark.common.policy import network_factory
from benchmark.common.utils import log_metrics, run_directory, write_json

# CLI options replace only explicitly supplied official PPO parameters.
PPO_ARGUMENTS = {
    "num_envs": int, "num_eval_envs": int, "unroll_length": int,
    "batch_size": int, "num_minibatches": int, "num_updates_per_batch": int,
    "num_evals": int, "num_resets_per_eval": int, "learning_rate": float, "discounting": float,
    "entropy_cost": float, "clipping_epsilon": float, "gae_lambda": float,
    "reward_scaling": float, "max_grad_norm": float,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", choices=TASKS, default=DEFAULT_TASK)
    p.add_argument("--obs-mode", choices=OBS_MODES, default="state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impl", choices=("jax", "warp"), help="Default: the task's native implementation")
    p.add_argument("--total-timesteps", type=int, help="Default: official task PPO training budget")
    p.add_argument("--output", help="New run directory (never overwrites an existing run)")
    p.add_argument("--run-evals", action=argparse.BooleanOptionalAction, default=True)
    for name, kind in PPO_ARGUMENTS.items():
        p.add_argument("--" + name.replace("_", "-"), type=kind)
    p.add_argument("--map-robot", choices=("full", "gripper"), default="full",
                   help="Robot visual parts: whole robot or manipulation hand/gripper")
    p.add_argument("--map-background", choices=("true", "false"), default="false",
                   help="Include task-unrelated scene objects; floors/walls/goals are always excluded")
    p.add_argument("--map-cache", default=".cache/maps")
    p.add_argument("--map-views", type=int, default=96)
    p.add_argument("--map-extra-views", type=int, default=512)
    p.add_argument("--dino-source", help="Local DINOv3 source (map cache creation only)")
    p.add_argument("--dino-weights", help="Local DINOv3 ViT-L/16 weights")
    return p


def train(args):
    start = time.monotonic()
    config = env_config(args.env_id, args.obs_mode, args.impl)
    params = ppo_config(config)
    for name in PPO_ARGUMENTS:
        value = getattr(args, name)
        if value is not None:
            nonnegative = name in ('entropy_cost', 'num_evals', 'num_resets_per_eval')
            if not math.isfinite(value) or value < 0 or (value == 0 and not nonnegative):
                raise ValueError(f"Invalid --{name.replace('_', '-')}: {value}")
            params[name] = value
    if args.total_timesteps is not None:
        if args.total_timesteps < 1:
            raise ValueError("--total-timesteps must be positive")
        params["num_timesteps"] = args.total_timesteps
    if params["batch_size"] * params["num_minibatches"] % params["num_envs"]:
        raise ValueError("Brax requires batch_size * num_minibatches to be divisible by num_envs")
    if args.obs_mode == "map":
        if args.map_views < 1 or args.map_extra_views < 0:
            raise ValueError("Map views must be positive and extra views nonnegative")
        from benchmark.common.dino import dino_config
        config["map"] = {"robot": args.map_robot, "background": args.map_background == "true",
                         "voxel_size": .015, "views": args.map_views, "extra_views": args.map_extra_views,
                         "cache": str(Path(args.map_cache).expanduser().resolve()),
                         "dino": dino_config(args.dino_source, args.dino_weights)}
    env = make_env(config, params["num_envs"])
    bank = env.bank if args.obs_mode == "map" else None
    factory = network_factory(args.obs_mode, params["network_factory"], bank)
    num_eval_envs = params.get("num_eval_envs", 128)
    eval_env = make_env(config, num_eval_envs) if args.run_evals else None
    output = run_directory(args.output, args.env_id, args.obs_mode, args.seed)
    metadata = {"format_version": 2, "algorithm": "brax_ppo", "seed": args.seed,
                "env_config": config, "ppo": params, "action_size": env.action_size,
                "observation_size": env.observation_size,
                "run_evals": args.run_evals, "setup_seconds": time.monotonic() - start,
                "versions": {name: version(name) for name in ("playground", "brax", "jax", "jaxlib", "mujoco", "mujoco-mjx",
                             "mujoco-warp", "warp-lang", "flax", "optax", "orbax-checkpoint", "numpy")},
                "sources": {name: json.loads(distribution(name).read_text("direct_url.json") or '{}')
                            for name in ('playground', 'brax')},
                "devices": [str(d) for d in jax.devices()]}
    if bank is not None:
        metadata.update(map_cache_paths=bank.paths, map_points=len(env.feature_ids), map_components=env.component_names)
        metadata["versions"].update({name: version(name) for name in ('torch', 'h5py', 'trimesh')})
    write_json(output / "config.json", metadata)
    print(f"Training {args.env_id}/{args.obs_mode}; results: {output}", flush=True)
    training_params = {key: value for key, value in params.items() if key != "network_factory"}
    ppo.train(
        environment=env, eval_env=eval_env, network_factory=factory,
        wrap_env_fn=wrapper.wrap_for_brax_training, vision=args.obs_mode == "rgb",
        seed=args.seed, save_checkpoint_path=str(output / "checkpoints"),
        progress_fn=lambda step, metrics: log_metrics(output / "train.csv", step, metrics),
        run_evals=args.run_evals, log_training_metrics=True, **training_params,
    )
    print(f"Saved run: {output}", flush=True)
    return output


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
