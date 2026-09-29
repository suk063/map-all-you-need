"""The fixed table-top dense-reward benchmark; no task discovery at runtime."""

import copy

import gymnasium as gym

TASKS = (
    "LiftPegUpright-v1", "PegInsertionSide-v1", "PickCube-v1",
    "PickCubeSO100-v1", "PickCubeWidowXAI-v1", "PickSingleYCB-v1",
    "PlaceSphere-v1", "PokeCube-v1", "PullCube-v1", "PullCubeTool-v1",
    "PushCube-v1", "PushT-v1", "RollBall-v1", "StackCube-v1",
    "TwoRobotPickCube-v1", "TwoRobotStackCube-v1",
)
OBS_MODES = ("state", "rgb", "rgbd", "dino", "map")
DEFAULT_CONTROL_MODE = "pd_ee_delta_pose"


class VisibleGoalWrapper(gym.Wrapper):
    """Show existing goal markers to sensors; leave physics and collisions untouched."""

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        base = self.unwrapped
        goals = [actor for actor in base._hidden_objects if "goal" in actor.name]
        for actor in goals:
            actor.show_visual()
        base._hidden_objects = [actor for actor in base._hidden_objects if "goal" not in actor.name]
        return base.get_obs(info), info


def shared_control_mode(mode):
    """ManiSkill 3.0.1 creates both robots with a shared controller name."""
    modes = list(mode.values()) if isinstance(mode, dict) else mode
    if isinstance(modes, (tuple, list)):
        if not modes or any(m != modes[0] for m in modes):
            raise ValueError("ManiSkill 3.0.1 requires the same controller for both robots at creation")
        mode = modes[0]
    if not isinstance(mode, str):
        raise TypeError("A controller name is required")
    return mode


def env_config(env_id, obs_mode, control_mode=DEFAULT_CONTROL_MODE):
    if env_id not in TASKS or obs_mode not in OBS_MODES:
        raise ValueError(f"Unsupported task/observation: {env_id}/{obs_mode}")
    return {"env_id": env_id, "env_kwargs": {
        "obs_mode": {"dino": "rgb", "map": "none"}.get(obs_mode, obs_mode), "control_mode": shared_control_mode(control_mode),
        "reward_mode": "dense", "sim_backend": "physx_cuda",
    }}


def make_env(config, num_envs=1, evaluation=False, map_bank=None):
    import mani_skill.envs  # noqa: F401  Registers ManiSkill environments.
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    if config["env_id"] not in TASKS:
        raise ValueError(f"Not a supported dense-reward task: {config['env_id']}")
    kwargs = copy.deepcopy(config["env_kwargs"])
    kwargs["control_mode"] = shared_control_mode(kwargs["control_mode"])
    if isinstance(kwargs.get("robot_uids"), list):
        kwargs["robot_uids"] = tuple(kwargs["robot_uids"])
    if kwargs["obs_mode"] not in ("state", "rgb", "rgbd", "none"):
        raise ValueError(f"Unsupported observation mode: {kwargs['obs_mode']}")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if kwargs["sim_backend"] == "physx_cpu" and num_envs != 1:
        raise ValueError("CPU evaluation uses --num-envs 1")
    kwargs.update(num_envs=num_envs, reward_mode="dense")
    if evaluation:
        kwargs["reconfiguration_freq"] = 1
    env = gym.make(config["env_id"], **kwargs)
    if isinstance(env.unwrapped.single_action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if config.get("show_goal", False):
        env = VisibleGoalWrapper(env)
    mapper = None
    if "map" in config:
        from benchmark.common.mapping import MapObservationWrapper
        mapper = MapObservationWrapper(env, config["map"], map_bank)
        env = mapper
    # Full horizons allow object reconfiguration on reset (notably YCB tasks).
    vector = ManiSkillVectorEnv(
        env, auto_reset=True, ignore_terminations=True, record_metrics=True,
    )
    vector.map_wrapper = mapper
    vector.map_bank = mapper.bank if mapper is not None else None
    return vector


def resolved_config(config, env):
    """Freeze defaults too, so a checkpoint describes its complete environment."""
    result = copy.deepcopy(config)
    result["env_kwargs"].update(
        robot_uids=env.unwrapped.robot_uids,
        control_mode=env.unwrapped.control_mode,
        max_episode_steps=(config["env_kwargs"].get("max_episode_steps")
                           or gym.spec(config["env_id"]).max_episode_steps),
    )
    return result


def action_spec(env):
    original = env.unwrapped.single_action_space
    parts = original.spaces.items() if isinstance(original, gym.spaces.Dict) else [(None, original)]
    return [{"key": key, "size": int(space.shape[0]),
             "low": space.low.tolist(), "high": space.high.tolist()}
            for key, space in parts]
