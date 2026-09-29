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
from benchmark.common.policy import (
    latest_checkpoint,
    network_factory,
    resolve_map_caches,
)
from benchmark.common.utils import log_metrics, run_directory, write_json
from benchmark.rl.resume import (
    checkpoint_schedule,
    resume_budget,
    validate_resume_overrides,
)

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
    p.add_argument("--env-id", choices=TASKS)
    p.add_argument("--obs-mode", choices=OBS_MODES)
    p.add_argument("--seed", type=int)
    p.add_argument("--impl", choices=("jax", "warp"), help="Default: the task's native implementation")
    p.add_argument("--total-timesteps", type=int, help="Default: official task PPO training budget")
    p.add_argument("--output", help="New run directory (never overwrites an existing run)")
    p.add_argument("--resume", help="Continue the latest valid checkpoint in a run; optimizer/RNG restart")
    p.add_argument("--checkpoint-steps", type=int, help="Approximate maximum steps between checkpoint epochs")
    p.add_argument("--run-evals", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=None,
                   help="Override observation normalization (map observations require false)")
    for name, kind in PPO_ARGUMENTS.items():
        p.add_argument("--" + name.replace("_", "-"), type=kind)
    p.add_argument("--map-robot", choices=("full", "gripper"),
                   help="Robot visual parts: whole robot or manipulation hand/gripper")
    p.add_argument("--map-background", choices=("true", "false"),
                   help="Include task-unrelated scene objects; floors/walls/Aloha tables are always excluded")
    p.add_argument("--map-cache")
    p.add_argument("--map-views", type=int)
    p.add_argument("--map-extra-views", type=int)
    p.add_argument("--dino-source", help="Local DINOv3 source (map cache creation only)")
    p.add_argument("--dino-weights", help="Local DINOv3 ViT-L/16 weights")
    return p


def train(args):
    start = time.monotonic()
    restore_params, source_checkpoint, source_root = None, None, None
    resume_base_steps = 0
    if args.resume:
        source_root = Path(args.resume).expanduser().resolve()
        saved = json.loads((source_root / "config.json").read_text())
        if saved.get("format_version") not in (2, 3):
            raise ValueError("Unsupported checkpoint format")
        validate_resume_overrides(args, saved)
        saved = resolve_map_caches(source_root, saved)
        source_checkpoint, restore_params = latest_checkpoint(source_root)
        target_timesteps, resume_base_steps, remaining = resume_budget(saved, source_checkpoint)
        if remaining == 0:
            print(f"Target already reached at saved step {resume_base_steps}: {source_root}", flush=True)
            return source_root
        config, params = saved["env_config"], saved["ppo"]
        args.env_id, args.obs_mode, args.seed = config["env_id"], config["obs_mode"], saved["seed"]
        params["num_timesteps"] = remaining
    else:
        defaults = {"env_id": DEFAULT_TASK, "obs_mode": "state", "seed": 0, "map_robot": "full",
                    "map_background": "false", "map_cache": ".cache/maps", "map_views": 96, "map_extra_views": 512}
        for name, value in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        config = env_config(args.env_id, args.obs_mode, args.impl)
        params = ppo_config(config)
        target_timesteps = args.total_timesteps if args.total_timesteps is not None else params["num_timesteps"]
    if args.normalize_observations is not None:
        if args.resume and args.normalize_observations != params['normalize_observations']:
            raise ValueError('--normalize-observations is incompatible with the resumed run')
        if args.obs_mode == 'map' and args.normalize_observations:
            raise ValueError('Map observations cannot be normalized')
        params['normalize_observations'] = args.normalize_observations
    for name in PPO_ARGUMENTS:
        value = getattr(args, name)
        if value is not None:
            nonnegative = name in ('entropy_cost', 'num_evals', 'num_resets_per_eval')
            if not math.isfinite(value) or value < 0 or (value == 0 and not nonnegative):
                raise ValueError(f"Invalid --{name.replace('_', '-')}: {value}")
            params[name] = value
    if args.total_timesteps is not None and not args.resume:
        if args.total_timesteps < 1:
            raise ValueError("--total-timesteps must be positive")
        params["num_timesteps"] = args.total_timesteps
    checkpoint_interval = None
    if args.checkpoint_steps is not None:
        if args.num_evals is not None:
            raise ValueError("Use either --checkpoint-steps or --num-evals")
        params["num_evals"], checkpoint_interval = checkpoint_schedule(params, args.checkpoint_steps)
    if params["batch_size"] * params["num_minibatches"] % params["num_envs"]:
        raise ValueError("Brax requires batch_size * num_minibatches to be divisible by num_envs")
    if args.obs_mode == "map" and not args.resume:
        if args.map_views < 1 or args.map_extra_views < 0:
            raise ValueError("Map views must be positive and extra views nonnegative")
        from benchmark.common.dino import dino_config
        from benchmark.common.mapping import DEFAULT_VOXEL_SIZE
        config["map"] = {"robot": args.map_robot, "background": args.map_background == "true",
                         "voxel_size": DEFAULT_VOXEL_SIZE, "views": args.map_views, "extra_views": args.map_extra_views,
                         "cache": str(Path(args.map_cache).expanduser().resolve()),
                         "dino": dino_config(args.dino_source, args.dino_weights)}
    env = make_env(config, params["num_envs"])
    bank = env.bank if args.obs_mode == "map" else None
    factory = network_factory(args.obs_mode, params["network_factory"], bank)
    num_eval_envs = params.get("num_eval_envs", 128)
    # Pinned Brax constructs/reset-compiles Evaluator even with run_evals=False.
    # Its reset batch must match its own renderer; no eval rollouts are enabled.
    eval_env = make_env(config, num_eval_envs) if args.run_evals or args.obs_mode == 'rgb' else None
    output = run_directory(args.output, args.env_id, args.obs_mode, args.seed)
    metadata = {"format_version": 3, "algorithm": "brax_ppo", "seed": args.seed,
                "env_config": config, "ppo": params, "action_size": env.action_size,
                "target_timesteps": target_timesteps, "resume_base_steps": resume_base_steps,
                "checkpoint_steps": args.checkpoint_steps, "checkpoint_interval_steps": checkpoint_interval,
                "observation_size": env.observation_size,
                "run_evals": args.run_evals, "setup_seconds": time.monotonic() - start,
                "versions": {name: version(name) for name in ("playground", "brax", "jax", "jaxlib", "mujoco", "mujoco-mjx",
                             "mujoco-warp", "warp-lang", "flax", "optax", "orbax-checkpoint", "numpy")},
                "sources": {name: json.loads(distribution(name).read_text("direct_url.json") or '{}')
                            for name in ('playground', 'brax')},
                "devices": [str(d) for d in jax.devices()]}
    if source_checkpoint is not None:
        if env.action_size != saved["action_size"] or json.loads(json.dumps(env.observation_size)) != saved["observation_size"]:
            raise ValueError("Environment/network shapes are incompatible with the resumed run")
        metadata["resume"] = {"run_directory": str(source_root), "checkpoint": str(source_checkpoint),
                              "checkpoint_steps": int(source_checkpoint.name),
                              "restored": ["normalizer", "actor", "critic"],
                              "optimizer_restored": False, "rng_restored": False}
    if hasattr(env, "render_metadata"):
        metadata["renderer"] = env.render_metadata
    if bank is not None:
        metadata.update(map_cache_paths=bank.paths, map_points=len(env.feature_ids), map_components=env.component_names)
        metadata["versions"].update({name: version(name) for name in ('torch', 'h5py', 'trimesh')})
    write_json(output / "config.json", metadata)
    print(f"Training {args.env_id}/{args.obs_mode}; results: {output}", flush=True)
    training_params = {key: value for key, value in params.items() if key != "network_factory"}
    # In pinned Brax `vision` only forbids action_repeat != 1. Our deferred
    # renderer runs after EpisodeWrapper, including PushCube's native repeat=4.
    legacy_vision = args.obs_mode == "rgb" and "rgb" not in config
    ppo.train(
        environment=env, eval_env=eval_env, network_factory=factory,
        wrap_env_fn=wrapper.wrap_for_brax_training, vision=legacy_vision,
        seed=args.seed, save_checkpoint_path=str(output / "checkpoints"),
        restore_params=restore_params, restore_value_fn=True,
        progress_fn=lambda step, metrics: log_metrics(output / "train.csv", resume_base_steps + step, metrics),
        run_evals=args.run_evals, log_training_metrics=True, **training_params,
    )
    print(f"Saved run: {output}", flush=True)
    return output


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
