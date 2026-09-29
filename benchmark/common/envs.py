"""Official manipulation tasks and their unchanged environment/PPO defaults."""

from ml_collections import ConfigDict
from mujoco_playground import manipulation, registry
from mujoco_playground._src.wrapper import Wrapper
from mujoco_playground.config import manipulation_params

TASKS = tuple(manipulation.ALL_ENVS)
OBS_MODES = ("state", "rgb", "map")
DEFAULT_TASK = "PandaPickCubeCartesian"
GOAL_BODIES = ("mocap_target", "goal")


class NonVisionWrapper(Wrapper):
    """The pinned upstream wrapper detects render hooks even when vision=False."""

    def __getattr__(self, name):
        if name == "defer_rendering":
            raise AttributeError(name)
        return super().__getattr__(name)


def env_config(env_id=DEFAULT_TASK, obs_mode="state", impl=None):
    if env_id not in TASKS or obs_mode not in OBS_MODES:
        raise ValueError(f"Unsupported task/observation: {env_id}/{obs_mode}")
    if obs_mode == "rgb" and env_id != DEFAULT_TASK:
        raise ValueError(f"Official RGB observations are supported only by {DEFAULT_TASK}")
    config = registry.get_default_config(env_id)
    if impl is not None:
        if "impl" not in config:
            raise ValueError(f"{env_id} does not expose an implementation override")
        config.impl = impl
    if obs_mode == "rgb":
        config.vision = True
    return {"env_id": env_id, "obs_mode": obs_mode, "goal_markers": True,
            "environment": config.to_dict()}


def ppo_config(config):
    factory = (manipulation_params.brax_vision_ppo_config if config["obs_mode"] == "rgb"
               else manipulation_params.brax_ppo_config)
    params = factory(config["env_id"], config["environment"].get("impl")).to_dict()
    if config["obs_mode"] == "map":
        params.update(num_envs=8, num_eval_envs=8, batch_size=1, normalize_observations=False)
        params["network_factory"] = {}  # Map actor and critic have their own encoders.
    return params


def make_env(config, num_envs=1):
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    # Validate saved configs too; there is no legacy backend/checkpoint adapter.
    env_config(config["env_id"], config["obs_mode"])
    if config.get("goal_markers") is False:
        raise ValueError("This run hides goal markers; use its saved source or retrain with native goals")
    values = config["environment"]
    if config["obs_mode"] == "rgb":
        # MuJoCo distinguishes a single (H, W) tuple from a list of per-camera sizes.
        values = {**values, "vision_config": {**values["vision_config"], "nworld": num_envs,
                  "cam_res": tuple(values["vision_config"]["cam_res"])}}
    env = registry.load(config["env_id"], config=ConfigDict(values))
    if config["obs_mode"] != "rgb" and hasattr(env, "defer_rendering"):
        env = NonVisionWrapper(env)
    if config["obs_mode"] == "map":
        from benchmark.common.mapping import MapObservationWrapper
        env = MapObservationWrapper(env, config["env_id"], config["map"])
    return env
