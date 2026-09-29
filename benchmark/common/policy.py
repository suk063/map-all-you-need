"""Official Brax networks and a small map network factory/checkpoint loader."""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks_vision
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen as nn

from benchmark.common.points import MapEncoder


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


def network_factory(mode, settings, bank=None):
    if mode == "state":
        return functools.partial(ppo_networks.make_ppo_networks, **settings)
    if mode == "rgb":
        return functools.partial(networks_vision.make_ppo_networks_vision, **settings)
    if mode != "map" or bank is None:
        raise ValueError("Map networks require their immutable feature bank")
    features = bank.features

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
        return self.inference(obs, jax.random.PRNGKey(0))[0]


def load_policy(run_directory):
    root = Path(run_directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("--checkpoint must be a Playground run directory (legacy .pt files are unsupported)")
    metadata = json.loads((root / "config.json").read_text())
    if metadata.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format; retrain with MuJoCo Playground")
    paths = sorted((p for p in (root / "checkpoints").glob("*") if p.is_dir() and p.name.isdigit()),
                   key=lambda p: int(p.name))
    if not paths:
        raise ValueError(f"No Brax checkpoint in {root}")
    bank = None
    if metadata["env_config"]["obs_mode"] == "map":
        from benchmark.common.mapping import FeatureBank
        bank = FeatureBank()
        for path in metadata["map_cache_paths"]:
            bank.add(path)
    factory = network_factory(metadata["env_config"]["obs_mode"], metadata["ppo"]["network_factory"], bank)
    # Persist actual observation shapes ourselves: the pinned Brax trainer writes
    # normalizer shapes (only the final axis) into its network config at save time.
    size = metadata["observation_size"]
    if isinstance(size, dict):
        size = {key: (value,) if isinstance(value, int) else tuple(value) for key, value in size.items()}
    preprocess = running_statistics.normalize if metadata["ppo"]["normalize_observations"] else types.identity_observation_preprocessor
    nets = factory(size, metadata["action_size"], preprocess_observations_fn=preprocess)
    inference = ppo_networks.make_inference_fn(nets)(checkpoint.load(paths[-1]), deterministic=True)
    return Policy(metadata, jax.jit(inference), paths[-1])
