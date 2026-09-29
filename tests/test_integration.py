import subprocess
import sys

import pytest

from benchmark.common.envs import TASKS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("env_id", TASKS)
def test_task_all_observations(env_id):
    subprocess.run([sys.executable, "-m", "tests.smoke", "env", "--env-id", env_id],
                   check=True, timeout=180)


@pytest.mark.parametrize("mode", ["state", "rgb", "rgbd"])
def test_training_and_evaluation(tmp_path, mode):
    subprocess.run([sys.executable, "-m", "tests.smoke", "train", "--obs-mode", mode,
                    "--output", str(tmp_path / mode)], check=True, timeout=300)


@pytest.mark.parametrize("env_id,mode,view", [
    ("PegInsertionSide-v1", "rgb", "external"),
    ("TwoRobotPickCube-v1", "rgbd", "wrist"),
])
def test_selected_view_training_and_evaluation(tmp_path, env_id, mode, view):
    subprocess.run([sys.executable, "-m", "tests.smoke", "train", "--env-id", env_id,
                    "--obs-mode", mode, "--view", view, "--output", str(tmp_path / view)],
                   check=True, timeout=300)


@pytest.mark.parametrize("env_id", TASKS)
def test_map_task(env_id):
    subprocess.run([sys.executable, "-m", "tests.new_smoke", "env", "--env-id", env_id,
                    "--map-robot", "gripper", "--map-background", "none"], check=True, timeout=900)


@pytest.mark.parametrize("mode,flags", [
    ("rgb", ["--no-state-input"]), ("rgbd", ["--no-state-input"]),
    ("dino", ["--state-input"]), ("dino", ["--no-state-input"]),
    ("map", ["--map-robot", "full", "--map-background", "table"]),
    ("map", ["--map-robot", "gripper", "--map-background", "none"]),
])
def test_new_training(tmp_path, mode, flags):
    subprocess.run([sys.executable, "-m", "tests.new_smoke", "train", "--obs-mode", mode,
                    *flags, "--output", str(tmp_path / mode)], check=True, timeout=1800)


def test_map_bc_alignment(tmp_path):
    subprocess.run([sys.executable, "-m", "tests.new_smoke", "align", "--map-robot", "gripper",
                    "--map-background", "none", "--output", str(tmp_path)], check=True, timeout=300)
