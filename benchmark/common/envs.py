"""Official manipulation tasks/PPO defaults with task-specific goal visibility."""

import numpy as np
from ml_collections import ConfigDict
from mujoco import mjx
from mujoco_playground import manipulation, registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda.pick_cartesian import (
    PandaPickCubeCartesian,
)
from mujoco_playground._src.wrapper import Wrapper
from mujoco_playground.config import manipulation_params

from benchmark.common.geometry import geometry_config, validate_geometry

TASKS = tuple(manipulation.ALL_ENVS)
OBS_MODES = ("state", "rgb", "map")
DEFAULT_TASK = "PandaPickCubeCartesian"
GOAL_BODIES = ("mocap_target", "goal")
GOAL_TASKS = frozenset({
    "AlohaHandOver", "PandaPickCube", "PandaPickCubeOrientation",
    "PandaPickCubeCartesian", "PandaOpenCabinet", "PandaRobotiqPushCube", "LeapCubeReorient",
})


class GoalVisibleCartesian(PandaPickCubeCartesian):
    """Show the native marker at the target in RGB, without changing physics."""

    def render_state(self, state):
        # This task has one unrotated goal box with zero local offset.
        geom = int(self.mj_model.body("mocap_target").geomadr[0])
        data = state.data.replace(geom_xpos=state.data.geom_xpos.at[..., geom, :].set(state.info["target_pos"]))
        rendered = super().render_state(state.replace(data=data))
        return state.replace(obs=rendered.obs)


class NonVisionWrapper(Wrapper):
    """The pinned upstream wrapper detects render hooks even when vision=False."""

    def __getattr__(self, name):
        if name == "defer_rendering":
            raise AttributeError(name)
        return super().__getattr__(name)


def env_config(env_id=DEFAULT_TASK, obs_mode="state", impl=None):
    if env_id not in TASKS or obs_mode not in OBS_MODES:
        raise ValueError(f"Unsupported task/observation: {env_id}/{obs_mode}")
    config = registry.get_default_config(env_id)
    if impl is not None:
        if "impl" not in config:
            raise ValueError(f"{env_id} does not expose an implementation override")
        config.impl = impl
    result = {"env_id": env_id, "obs_mode": obs_mode, "goal_markers": "task",
              "environment": config.to_dict()}
    if obs_mode == "rgb":
        from benchmark.common.rgb import rgb_config
        result['rgb'] = rgb_config(env_id)
    elif obs_mode == 'map':
        result['map_geometry'] = geometry_config(env_id)
    return result


def ppo_config(config):
    params = manipulation_params.brax_ppo_config(config["env_id"], config["environment"].get("impl")).to_dict()
    if config['obs_mode'] == 'rgb':
        params.update(num_envs=128, num_eval_envs=8, batch_size=16, normalize_observations=False)
        params['network_factory'] = manipulation_params.brax_vision_ppo_config(DEFAULT_TASK).network_factory.to_dict()
    if config["obs_mode"] == "map":
        params.update(num_envs=8, num_eval_envs=8, batch_size=1, normalize_observations=False)
        params["network_factory"] = {}  # Map actor and critic have their own encoders.
    return params


def hide_goal_markers(env, vision=False):
    """Hide unused markers without changing bodies, sensors, or collisions."""
    model = env.mj_model
    bodies = [body for body in range(model.nbody) if model.body(body).name in GOAL_BODIES]
    geoms = np.isin(model.geom_bodyid, bodies)
    if not geoms.any():
        return
    model.geom_group[geoms] = 5
    model.geom_rgba[geoms, 3] = 0
    model.geom_matid[geoms] = -1
    env._mjx_model = mjx_env.put_model(model, impl=env.mjx_model.impl.value)
    if vision:
        env._rc = mjx.create_render_context(mjm=model, **env._config.vision_config.to_dict())
        env._rc_pytree = env._rc.pytree()


def make_env(config, num_envs=1):
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    # Missing `rgb` marks the old format-2 official Cartesian vision behavior.
    env_config(config["env_id"], config["obs_mode"])
    legacy_vision = config["obs_mode"] == "rgb" and 'rgb' not in config
    legacy_hidden = legacy_vision and config['env_id'] == DEFAULT_TASK and config.get('goal_markers') is False
    if config.get("goal_markers") != "task" and not legacy_hidden:
        raise ValueError("Run uses different goal visibility; use its original source or retrain")
    validate_geometry(config)
    if legacy_vision and config['env_id'] != DEFAULT_TASK:
        raise ValueError(f'Legacy native RGB is supported only by {DEFAULT_TASK}')
    values = config["environment"]
    if legacy_vision:
        # MuJoCo distinguishes a single (H, W) tuple from a list of per-camera sizes.
        values = {**values, "vision_config": {**values["vision_config"], "nworld": num_envs,
                  "cam_res": tuple(values["vision_config"]["cam_res"])}}
    env = (GoalVisibleCartesian(config=ConfigDict(values)) if legacy_vision and not legacy_hidden
           else registry.load(config["env_id"], config=ConfigDict(values)))
    if legacy_hidden or config['env_id'] not in GOAL_TASKS:
        hide_goal_markers(env, vision=legacy_vision)
    if config["obs_mode"] == "rgb" and not legacy_vision:
        from benchmark.common.rgb import RGBObservationWrapper
        env = RGBObservationWrapper(env, config['rgb'], num_envs)
    elif not legacy_vision and hasattr(env, "defer_rendering"):
        env = NonVisionWrapper(env)
    if config["obs_mode"] == "map":
        from benchmark.common.mapping import MapObservationWrapper
        env = MapObservationWrapper(env, config["env_id"], config["map"])
    return env
