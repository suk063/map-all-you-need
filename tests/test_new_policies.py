from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from benchmark.bc.data import DemoDataset
from benchmark.bc.map_data import MapDemoDataset, collate_maps
from benchmark.common.envs import env_config
from benchmark.common.mapping import FeatureBank, batch_observations, surface_points
from benchmark.common.points import MapEncoder
from benchmark.common.policy import (
    Policy,
    load_policy,
    observation_spec,
    prepare_observation,
    save_policy,
)
from tests.test_core import demo_fixture, visual_obs


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("mode", ["rgb", "rgbd"])
def test_image_only_never_reads_state(tmp_path, mode):
    path, metadata, _, actions, raw = demo_fixture(tmp_path, mode)
    spec = observation_spec(raw, mode, state_input=False)
    assert spec["state_paths"] == [] and spec["state_dim"] == 0
    del raw["agent"], raw["extra"]
    with h5py.File(path, "a") as f:
        for trajectory in f.values():
            del trajectory["obs/agent"], trajectory["obs/extra"]
    dataset = DemoDataset(path, metadata, spec, actions)
    try:
        sample, _ = dataset[0]
        expected = prepare_observation(raw, spec, "cpu")
        assert "state" not in sample
        for key in sample:
            torch.testing.assert_close(sample[key], expected[key][0])
        policy = Policy(spec, actions, env_config("PickCube-v1", mode))
        file = tmp_path / "policy.pt"
        save_policy(file, policy, {})
        torch.testing.assert_close(policy.act(raw), load_policy(file).act(raw), rtol=0, atol=0)
    finally:
        dataset.close()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 384)

    def forward_features(self, x):
        assert x.shape[-2:] == (128, 128)
        patches = F.adaptive_avg_pool2d(x, (8, 8)).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": self.project(patches)}


@pytest.mark.parametrize("state_input", [True, False])
def test_dino_frozen_cache_spatial_features_and_checkpoint(monkeypatch, tmp_path, state_input):
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: TinyBackbone())
    raw = visual_obs()
    spec = observation_spec(raw, "dino", state_input=state_input, dino={"source": "local", "model": "test"})
    actions = [{"key": None, "size": 2, "low": [-1., -1.], "high": [1., 1.]}]
    policy = Policy(spec, actions, env_config("PickCube-v1", "dino"), pretrained=False)
    policy.train()
    assert not policy.encoder.image.backbone.training
    prepared = policy.prepare(raw)
    assert prepared["dino"].shape == (1, 2 * 64, 384)
    assert "rgb" not in prepared
    uncached = prepare_observation(raw, spec, "cpu")
    torch.testing.assert_close(policy(prepared), policy(uncached))
    policy(prepared).square().sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in policy.encoder.image.backbone.parameters())
    assert policy.encoder.image.patch.weight.grad.abs().sum() > 0
    path = tmp_path / "dino.pt"
    save_policy(path, policy, {})
    torch.testing.assert_close(load_policy(path).act(raw), policy.act(raw), rtol=0, atol=0)


@pytest.fixture
def point_bank(tmp_path):
    torch.manual_seed(2)
    path = tmp_path / "points.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = 1
        f["xyz"] = np.zeros((512, 3), dtype=np.float32)
        f["features"] = torch.randn(512, 1024).half().numpy()
    return FeatureBank([path])


def point_obs(n=48):
    return {"xyz": torch.rand(1, n, 3), "feature_ids": torch.arange(n)[None]}


def test_relative_only_global_token_and_gradients(point_bank):
    encoder = MapEncoder(point_bank)
    raw = point_obs()
    expected = encoder(raw)
    assert expected.shape == (1, 256)
    assert sum(p.numel() for p in encoder.parameters()) < 1_000_000
    shifted = {**raw, "xyz": raw["xyz"] + torch.tensor([2., -3., .7])}
    torch.testing.assert_close(encoder(shifted), expected, rtol=2e-5, atol=2e-5)
    moved = {**raw, "xyz": raw["xyz"].clone()}
    moved["xyz"][:, :10] += .3
    assert not torch.allclose(encoder(moved), expected, atol=1e-5)
    expected.square().mean().backward()
    assert encoder.input.weight.grad.abs().sum() > 0
    assert encoder.global_block.position[0].weight.grad.abs().sum() > 0


def test_point_padding_and_snapshot_batching(point_bank):
    encoder = MapEncoder(point_bank).eval()
    small, big = point_obs(3), point_obs(48)
    batch = batch_observations([small, big])
    assert batch["feature_ids"][0, 3:].eq(-1).all()
    expected = torch.cat([encoder(small), encoder(big)])
    torch.testing.assert_close(encoder(batch), expected, rtol=2e-5, atol=2e-5)
    collated, actions = collate_maps([({k: v[0] for k, v in small.items()}, torch.zeros(2)),
                                    ({k: v[0] for k, v in big.items()}, torch.ones(2))])
    torch.testing.assert_close(collated["xyz"], batch["xyz"])
    assert actions.shape == (2, 2)


def test_translation_invariance_with_planar_sampling_ties(point_bank):
    xy = torch.cartesian_prod(torch.linspace(-.9, .9, 18), torch.linspace(-.9, .9, 18))
    xyz = F.pad(xy, (0, 1))[None]
    obs = {"xyz": xyz, "feature_ids": torch.arange(xyz.shape[1])[None]}
    encoder = MapEncoder(point_bank).eval()
    torch.testing.assert_close(encoder(obs), encoder({**obs, "xyz": xyz + torch.tensor([2., -3., .7])}), rtol=2e-5, atol=2e-5)


def test_map_policy_checkpoint_translation_invariance(tmp_path, point_bank):
    raw = point_obs()
    config = {"robot": "gripper", "background": "none", "voxel_size": .015}
    spec = observation_spec(raw, "map", map_config=config)
    actions = [{"key": None, "size": 2, "low": [-1., -1.], "high": [1., 1.]}]
    policy = Policy(spec, actions, env_config("PickCube-v1", "map"), point_bank).eval()
    path = tmp_path / "policy.pt"
    save_policy(path, policy, {})
    restored = load_policy(path)
    torch.testing.assert_close(restored.act(raw), policy.act(raw), rtol=0, atol=0)
    torch.testing.assert_close(policy.act({**raw, "xyz": raw["xyz"] + 3}), policy.act(raw), atol=1e-6, rtol=1e-5)
    assert spec["state_paths"] == [] and spec["cameras"] == []
    with pytest.raises(ValueError, match="forbids"):
        observation_spec(raw, "map", state_input=True, map_config=config)


def test_voxels_keep_one_surface_point_and_tabletop_only():
    import trimesh
    mesh = trimesh.creation.box(extents=[.3, .3, .1])
    xyz, _ = surface_points(mesh, .015)
    assert len(np.unique(np.floor(xyz / .015).astype(int), axis=0)) == len(xyz)
    top, _ = surface_points(mesh, .015, tabletop=True)
    np.testing.assert_allclose(top[:, 2], .05)
    assert len(top) < len(xyz)


def test_map_demo_requires_recorded_states(tmp_path):
    path, metadata, _, actions, _ = demo_fixture(tmp_path)
    with pytest.raises(ValueError, match="env_states"):
        MapDemoDataset(path, metadata, SimpleNamespace(), actions)


@pytest.mark.parametrize("length,nan", [(2, False), (3, True)])
def test_invalid_map_state_trajectories_fail_before_replay(tmp_path, length, nan):
    path, metadata, _, actions, _ = demo_fixture(tmp_path)
    with h5py.File(path, "a") as f:
        for trajectory in f.values():
            states = trajectory.create_group("env_states")
            data = np.zeros((length, 13), np.float32)
            if nan:
                data[0, 0] = np.nan
            states.create_group("actors").create_dataset("cube", data=data)
            states.create_group("articulations").create_dataset("panda", data=np.zeros((3, 31)))
    with pytest.raises(ValueError, match="finite T\\+1"):
        MapDemoDataset(path, metadata, SimpleNamespace(), actions)
