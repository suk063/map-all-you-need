"""Official Brax networks and a small map network factory/checkpoint loader."""

import copy
import functools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from benchmark.common.geometry import validate_geometry


def network_factory(mode, settings, bank=None):
    import jax.numpy as jnp
    from brax.training import distribution, networks, types
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import networks_vision
    from flax import linen as nn

    from benchmark.common.points import MapEncoder

    if mode == "state":
        return functools.partial(ppo_networks.make_ppo_networks, **settings)
    if mode == "rgb":
        return functools.partial(networks_vision.make_ppo_networks_vision, **settings)
    if mode != "map" or bank is None:
        raise ValueError("Map networks require their immutable feature bank")
    features = bank.features

    class MapHead(nn.Module):
        output_size: int
        value: bool = False

        @nn.compact
        def __call__(self, obs, bank):
            x = MapEncoder()(obs, bank)
            if self.value:
                x = nn.tanh(nn.Dense(256)(x))
            x = nn.Dense(self.output_size)(x)
            return x.squeeze(-1) if self.value else x

    def make_map_networks(observation_size, action_size,
                          preprocess_observations_fn=types.identity_observation_preprocessor):
        # Geometry and integer feature IDs must not be normalized.
        del preprocess_observations_fn
        action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
        dummy = {key: jnp.zeros((1, *shape)) for key, shape in observation_size.items()}

        def make_head(output_size, value=False):
            module = MapHead(output_size, value)
            return networks.FeedForwardNetwork(
                init=lambda key: module.init(key, dummy, features),
                apply=lambda processor_params, params, obs: module.apply(params, obs, features),
            )

        return ppo_networks.PPONetworks(
            policy_network=make_head(action_distribution.param_size),
            value_network=make_head(1, value=True),
            parametric_action_distribution=action_distribution,
        )

    return make_map_networks


@dataclass
class Policy:
    metadata: dict
    inference: object
    checkpoint: Path

    @property
    def env_config(self):
        return self.metadata["env_config"]

    def act(self, obs):
        import jax
        return self.inference(obs, jax.random.PRNGKey(0))[0]


def latest_checkpoint(root, *, load=None):
    """Return the newest loadable finalized PPO checkpoint, ignoring partial saves.

    `load` is an optional loader for lightweight storage checks/tests. Production
    callers use pinned Brax checkpoint.load, which validates Orbax's saved tree.
    Checkpoint names are attempt-local environment steps, never CSV progress.
    """
    if load is None:
        from brax.training.agents.ppo import checkpoint
        load = checkpoint.load
    root = Path(root).expanduser().resolve()
    directory = root / "checkpoints"
    paths = sorted((p for p in directory.iterdir() if p.is_dir() and p.name.isascii() and p.name.isdigit()),
                   key=lambda p: int(p.name), reverse=True) if directory.is_dir() else []
    failures = []
    for path in paths:
        try:
            params = load(path)
            if not isinstance(params, (tuple, list)) or len(params) != 3 or any(p is None for p in params):
                raise ValueError("expected normalizer, actor and critic parameters")
        except Exception as error:  # noqa: BLE001 - Orbax backends expose several corruption exception types.
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        if failures:
            warnings.warn("Skipped invalid checkpoints: " + "; ".join(failures), RuntimeWarning)
        return path, params
    detail = "; ".join(failures) or "no finalized numeric directories"
    raise ValueError(f"No valid Brax checkpoint in {root}: {detail}")


def resolve_map_caches(root, metadata):
    """Resolve a transferred run's complete map bundle without requiring DINO."""
    metadata = copy.deepcopy(metadata)
    bundle = Path(root).expanduser().resolve() / "map-cache"
    if metadata["env_config"]["obs_mode"] != "map" or not bundle.exists():
        return metadata
    paths = [bundle / Path(path).name for path in metadata["map_cache_paths"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing or not paths:
        raise ValueError("Incomplete bundled map cache: " + ", ".join(missing or [str(bundle)]))
    metadata["map_cache_paths"] = [str(path) for path in paths]
    metadata["env_config"]["map"].update(cache=str(bundle), cache_read_only=True)
    return metadata


def load_policy(run_directory):
    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("--checkpoint must be a Playground run directory (legacy .pt files are unsupported)")
    metadata = json.loads((root / "config.json").read_text())
    if metadata.get("format_version") not in (2, 3):
        raise ValueError("Unsupported checkpoint format; retrain with MuJoCo Playground")
    validate_geometry(metadata['env_config'])
    import jax
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    metadata = resolve_map_caches(root, metadata)
    checkpoint_path, params = latest_checkpoint(root)
    bank = None
    if metadata["env_config"]["obs_mode"] == "map":
        from benchmark.common.mapping import FeatureBank
        bank = FeatureBank()
        for cache_path in metadata["map_cache_paths"]:
            bank.add(cache_path)
    factory = network_factory(metadata["env_config"]["obs_mode"], metadata["ppo"]["network_factory"], bank)
    # Persist actual observation shapes ourselves: the pinned Brax trainer writes
    # normalizer shapes (only the final axis) into its network config at save time.
    size = metadata["observation_size"]
    if isinstance(size, dict):
        size = {key: (value,) if isinstance(value, int) else tuple(value) for key, value in size.items()}
    preprocess = running_statistics.normalize if metadata["ppo"]["normalize_observations"] else types.identity_observation_preprocessor
    nets = factory(size, metadata["action_size"], preprocess_observations_fn=preprocess)
    inference = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    return Policy(metadata, jax.jit(inference), checkpoint_path)
