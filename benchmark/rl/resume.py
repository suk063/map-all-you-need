"""Dependency-free validation and cumulative accounting for PPO attempts."""

import math
from pathlib import Path


def resume_budget(metadata, checkpoint):
    target = int(metadata.get("target_timesteps", metadata["ppo"]["num_timesteps"]))
    completed = int(metadata.get("resume_base_steps", 0)) + int(Path(checkpoint).name)
    if target < 1 or completed < 0:
        raise ValueError("Invalid saved training budget")
    return target, completed, max(0, target - completed)


def checkpoint_schedule(params, requested_steps):
    """Quantize checkpoint epochs to native PPO reset groups near the request.

    Brax rounds each epoch up. Choosing the epoch count from whole native groups
    avoids multiplying a large batch by an impossible sub-batch save frequency.
    Total rounding is less than one checkpoint interval. A native group larger
    than the request is the minimum interval.
    """
    if requested_steps < 1:
        raise ValueError("--checkpoint-steps must be positive")
    quantum = (params["batch_size"] * params["unroll_length"] * params["num_minibatches"]
               * params["action_repeat"] * max(params.get("num_resets_per_eval", 0), 1))
    groups = math.ceil(params["num_timesteps"] / quantum)
    groups_per_epoch = max(1, round(requested_steps / quantum))
    epochs = math.ceil(groups / groups_per_epoch)
    actual_groups_per_epoch = math.ceil(params['num_timesteps'] / (epochs * quantum))
    return epochs + 1, actual_groups_per_epoch * quantum


def validate_resume_overrides(args, metadata):
    """Only explicit identity overrides are checked; absent options inherit."""
    from benchmark.common.geometry import validate_geometry
    config = metadata["env_config"]
    validate_geometry(config)
    expected = {"env_id": config["env_id"], "obs_mode": config["obs_mode"], "seed": metadata["seed"],
                "impl": config["environment"].get("impl"),
                "total_timesteps": metadata.get("target_timesteps", metadata.get("ppo", {}).get("num_timesteps"))}
    if 'normalize_observations' in metadata.get('ppo', {}):
        expected['normalize_observations'] = metadata['ppo']['normalize_observations']
    if config["obs_mode"] == "map":
        mapping = config["map"]
        expected.update(map_robot=mapping.get("robot", "full"), map_background=str(mapping["background"]).lower(),
                        map_views=mapping["views"], map_extra_views=mapping["extra_views"],
                        map_cache=mapping["cache"], dino_source=mapping["dino"]["source"],
                        dino_weights=mapping["dino"]["weights"])
    for key, expected_value in expected.items():
        supplied = getattr(args, key, None)
        if supplied is None:
            continue
        if key in ("map_cache", "dino_source", "dino_weights"):
            supplied, expected_value = (str(Path(value).expanduser().resolve()) for value in (supplied, expected_value))
        if supplied != expected_value:
            raise ValueError(f"--{key.replace('_', '-')} is incompatible with the resumed run: {supplied!r} != {expected_value!r}")
