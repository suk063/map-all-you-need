import json

import h5py
import numpy as np
import pytest
import torch

from benchmark.bc.data import DemoDataset, read_metadata
from benchmark.common.envs import env_config, shared_control_mode
from benchmark.common.policy import (
    Policy,
    load_policy,
    observation_spec,
    prepare_observation,
    save_policy,
)
from benchmark.rl.train import compute_gae


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


def test_gae_uses_timeout_value_and_does_not_cross_resets():
    column = lambda x: torch.tensor(x, dtype=torch.float32)[:, None]
    advantages, returns = compute_gae(
        column([1, 100, 5]), column([2, 3, 4]), column([7, 9, 11]),
        torch.tensor([[False], [True], [False]]),
        torch.tensor([[True], [False], [False]]), 0.5, 1.0,
    )
    torch.testing.assert_close(advantages, column([2.5, 97, 6.5]))
    torch.testing.assert_close(returns, column([4.5, 100, 10.5]))


def test_gae_propagates_inside_episode():
    x = torch.tensor([[1.0], [2.0]])
    false = torch.zeros_like(x, dtype=torch.bool)
    advantages, _ = compute_gae(x, x * 0, x * 0, false, false, 0.5, 0.8)
    torch.testing.assert_close(advantages, torch.tensor([[1.8], [2.0]]))


def visual_obs():
    return {"agent": {"qpos": np.array([[1, 2]], dtype=np.float32), "controller": {}},
            "extra": {"is_grasped": np.array([True]), "goal_pos": np.array([[3, 4, 5]], dtype=np.float32)},
            "sensor_data": {key: {"rgb": np.full((1, 8, 8, 3), rgb, np.uint8),
                                   "depth": np.full((1, 8, 8, 1), 1500, np.uint16)}
                            for key, rgb in [("hand_camera", 255), ("base_camera", 0)]}}


def write_tree(group, tree):
    if isinstance(tree, dict):
        for k, v in tree.items():
            if isinstance(v, dict):
                write_tree(group.create_group(k), v)
            else:
                group.create_dataset(k, data=v)


@pytest.mark.parametrize("mode", ["state", "rgb", "rgbd"])
def test_policy_checkpoint_roundtrip(tmp_path, mode):
    obs = np.ones((1, 5), np.float32) if mode == "state" else visual_obs()
    spec = observation_spec(obs, mode)
    actions = [{"key": None, "size": 2, "low": [-1.0, -1.0], "high": [1.0, 1.0]}]
    policy = Policy(spec, actions, env_config("PickCube-v1", mode)).eval()
    path = tmp_path / "policy.pt"
    expected = policy.act(obs)
    save_policy(path, policy, {"algorithm": "bc", "seed": 0})
    restored = load_policy(path)
    torch.testing.assert_close(restored.act(obs), expected, rtol=0, atol=0)
    assert restored.env_config == policy.env_config
    if mode != "state":
        prepared = restored.prepare(obs)
        assert spec["cameras"] == ["base_camera", "hand_camera"]
        assert prepared["rgb"].shape == (1, 6, 128, 128)
        assert prepared["rgb"][0, :3].max() == 0
        assert prepared["rgb"][0, 3:].min() == 255
        assert ["extra", "is_grasped"] in spec["state_paths"]
        if mode == "rgbd":
            assert torch.all(prepared["depth"] == 1.5)


def demo_fixture(tmp_path, mode="state", multi=False):
    path = tmp_path / "trajectory.h5"
    obs = np.array([[1, 2], [3, 4], [999, 999]], dtype=np.float32)
    metadata = {"env_info": env_config("TwoRobotPickCube-v1" if multi else "PickCube-v1", mode),
                "episodes": [{"episode_id": 7}, {"episode_id": 2}]}
    action_parts = ([{"key": k, "size": 1, "low": [-1.0], "high": [1.0]} for k in ("left", "right")]
                    if multi else [{"key": None, "size": 2, "low": [-1.0, -1.0], "high": [1.0, 1.0]}])
    with h5py.File(path, "w") as data:
        for index in [7, 2]:
            group = data.create_group(f"traj_{index}")
            if mode == "state":
                group.create_dataset("obs", data=obs)
                example = obs[:1]
            else:
                example = visual_obs()
                def repeat(tree):
                    return {k: repeat(v) if isinstance(v, dict) else np.repeat(v, 3, axis=0) for k, v in tree.items()}
                write_tree(group.create_group("obs"), repeat(example))
            actions = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            if multi:
                action_group = group.create_group("actions")
                for i, key in enumerate(("left", "right")):
                    action_group.create_dataset(key, data=actions[:, i:i+1])
            else:
                group.create_dataset("actions", data=actions)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, metadata, observation_spec(example, mode), action_parts, example


@pytest.mark.parametrize("mode", ["state", "rgb", "rgbd"])
@pytest.mark.parametrize("multi", [False, True])
def test_demo_alignment_and_preprocessing(tmp_path, mode, multi):
    path, metadata, spec, actions, example = demo_fixture(tmp_path, mode, multi)
    dataset = DemoDataset(path, metadata, spec, actions)
    try:
        assert len(dataset) == 4
        obs, action = dataset[0]
        online = prepare_observation(example, spec, "cpu")
        for k in online:
            torch.testing.assert_close(obs[k], online[k][0], rtol=0, atol=0)
        torch.testing.assert_close(action, torch.tensor([0.1, 0.2]))
        torch.testing.assert_close(dataset[1][1], torch.tensor([0.3, 0.4]))
        torch.testing.assert_close(dataset[2][1], torch.tensor([0.1, 0.2]))
        if mode == "state":
            assert dataset[1][0]["state"].tolist() == [3, 4]
            assert dataset[2][0]["state"].tolist() == [1, 2]
    finally:
        dataset.close()


def test_bad_demos_fail_early(tmp_path):
    path, metadata, spec, actions, _ = demo_fixture(tmp_path)
    with pytest.raises(ValueError, match="requested"):
        read_metadata(path, "rgb")
    metadata["env_info"]["env_kwargs"]["obs_mode"] = "none"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="replay"):
        read_metadata(path, "state")
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="matching"):
        read_metadata(path, "state")
    with h5py.File(path, "a") as f:
        del f["traj_7/obs"]
    with pytest.raises(ValueError, match="missing obs"):
        DemoDataset(path, metadata, spec, actions)


@pytest.mark.parametrize("bad_field", ["obs", "actions"])
def test_bad_demo_lengths_and_action_dimensions(tmp_path, bad_field):
    path, metadata, spec, actions, _ = demo_fixture(tmp_path)
    with h5py.File(path, "a") as f:
        del f[f"traj_7/{bad_field}"]
        f["traj_7"].create_dataset(bad_field, data=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        DemoDataset(path, metadata, spec, actions)


def test_native_multi_robot_metadata(tmp_path):
    path, metadata, _, _, _ = demo_fixture(tmp_path, multi=True)
    controllers = {"panda_wristcam-0": "pd_joint_delta_pos", "panda_wristcam-1": "pd_joint_delta_pos"}
    metadata["env_info"]["env_kwargs"]["control_mode"] = controllers
    metadata["env_info"]["env_kwargs"]["robot_uids"] = ["panda_wristcam", "panda_wristcam"]
    for episode in metadata["episodes"]:
        episode["control_mode"] = controllers
    path.with_suffix(".json").write_text(json.dumps(metadata))
    _, config = read_metadata(path, "state")
    assert config["env_kwargs"]["control_mode"] == "pd_joint_delta_pos"
    with pytest.raises(ValueError, match="same controller"):
        shared_control_mode(["pd_joint_pos", "pd_joint_delta_pos"])


def test_bc_controller_selection_never_reinterprets_actions(tmp_path):
    path, metadata, _, _, _ = demo_fixture(tmp_path)
    assert read_metadata(path, "state", "pd_ee_delta_pose")[1]["env_kwargs"]["control_mode"] == "pd_ee_delta_pose"
    # Absolute and delta pose have the same action dimension but different semantics.
    with pytest.raises(ValueError, match="--target-control-mode pd_ee_pose"):
        read_metadata(path, "state", "pd_ee_pose")
    metadata["env_info"]["env_kwargs"]["control_mode"] = "pd_joint_delta_pos"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="--control-mode pd_joint_delta_pos"):
        read_metadata(path, "state", "pd_ee_delta_pose")
    _, config = read_metadata(path, "state", "pd_joint_delta_pos")
    assert config["env_kwargs"]["control_mode"] == "pd_joint_delta_pos"


def test_rl_demo_preserves_unclipped_actions(tmp_path):
    path, metadata, spec, actions, _ = demo_fixture(tmp_path)
    with h5py.File(path, "a") as f:
        f["traj_7/actions"][0] = [2.5, -3.0]
    dataset = DemoDataset(path, metadata, spec, actions)
    try:
        torch.testing.assert_close(dataset[0][1], torch.tensor([2.5, -3.0]))
    finally:
        dataset.close()


@pytest.mark.parametrize("mode", ["rgb", "rgbd"])
@pytest.mark.parametrize("view,expected", [
    ("external", ["base_camera"]),
    ("wrist", ["hand_camera"]),
    ("all", ["base_camera", "hand_camera"]),
])
def test_camera_view_input_and_checkpoint(tmp_path, mode, view, expected):
    obs = visual_obs()
    spec = observation_spec(obs, mode, view)
    assert spec["cameras"] == expected
    actions = [{"key": None, "size": 2, "low": [-1.0, -1.0], "high": [1.0, 1.0]}]
    policy = Policy(spec, actions, env_config("PegInsertionSide-v1", mode)).eval()
    prepared = policy.prepare(obs)
    assert prepared["rgb"].shape == (1, len(expected) * 3, 128, 128)
    if view == "external":
        assert torch.all(prepared["rgb"] == 0)
    elif view == "wrist":
        assert torch.all(prepared["rgb"] == 255)
    if mode == "rgbd":
        assert prepared["depth"].shape == (1, len(expected), 128, 128)
        assert torch.all(prepared["depth"] == 1.5)
    full = prepare_observation(obs, observation_spec(obs, mode), "cpu")
    torch.testing.assert_close(prepared["state"], full["state"])
    path = tmp_path / "policy.pt"
    save_policy(path, policy, {"algorithm": "bc", "view": view})
    loaded = load_policy(path)
    assert loaded.obs_spec == spec
    torch.testing.assert_close(loaded.act(obs), policy.act(obs), rtol=0, atol=0)


def test_both_robot_wrist_cameras_are_selected():
    obs = visual_obs()
    wrist = obs["sensor_data"].pop("hand_camera")
    obs["sensor_data"].update({f"panda_wristcam-{i}-hand_camera": wrist for i in (0, 1)})
    spec = observation_spec(obs, "rgbd", "wrist")
    assert spec["cameras"] == ["panda_wristcam-0-hand_camera", "panda_wristcam-1-hand_camera"]
    assert observation_spec(obs, "rgbd", "external")["cameras"] == ["base_camera"]
    assert prepare_observation(obs, spec, "cpu")["rgb"].shape == (1, 6, 128, 128)


def test_missing_views_and_state_view_fail_clearly():
    obs = visual_obs()
    del obs["sensor_data"]["hand_camera"]
    with pytest.raises(ValueError, match="No cameras.*wrist.*base_camera"):
        observation_spec(obs, "rgb", "wrist")
    obs = visual_obs()
    del obs["sensor_data"]["base_camera"]
    with pytest.raises(ValueError, match="No cameras.*external.*hand_camera"):
        observation_spec(obs, "rgbd", "external")
    with pytest.raises(ValueError, match="requires --obs-mode rgb or rgbd"):
        observation_spec(np.zeros((1, 5)), "state", "wrist")


@pytest.mark.parametrize("view,unused", [("wrist", "base_camera"), ("external", "hand_camera")])
def test_bc_does_not_read_unselected_cameras(tmp_path, view, unused):
    path, metadata, _, actions, obs = demo_fixture(tmp_path, "rgbd")
    spec = observation_spec(obs, "rgbd", view)
    with h5py.File(path, "a") as f:
        for name in f:
            del f[f"{name}/obs/sensor_data/{unused}"]
    dataset = DemoDataset(path, metadata, spec, actions)
    try:
        sample, _ = dataset[0]
        expected = prepare_observation(obs, spec, "cpu")
        for key in sample:
            torch.testing.assert_close(sample[key], expected[key][0])
    finally:
        dataset.close()
    with pytest.raises(ValueError, match="missing selected cameras"):
        DemoDataset(path, metadata, observation_spec(obs, "rgbd", "all"), actions)


@pytest.mark.parametrize("mode", ["state", "rgbd"])
def test_legacy_checkpoint_without_view_loads_as_all(tmp_path, mode):
    obs = np.ones((1, 5), np.float32) if mode == "state" else visual_obs()
    spec = observation_spec(obs, mode)
    actions = [{"key": None, "size": 2, "low": [-1.0, -1.0], "high": [1.0, 1.0]}]
    policy = Policy(spec, actions, env_config("PickCube-v1", mode)).eval()
    path = tmp_path / "legacy.pt"
    save_policy(path, policy, {"algorithm": "ppo"})
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["obs_spec"]["view"]
    del checkpoint["obs_spec"]["state_input"]
    torch.save(checkpoint, path)
    loaded = load_policy(path)
    assert loaded.obs_spec == observation_spec(obs, mode)
    assert loaded.obs_spec["view"] == "all"
    torch.testing.assert_close(loaded.act(obs), policy.act(obs), rtol=0, atol=0)
