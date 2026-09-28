import subprocess
import sys

import pytest

from benchmark.common.envs import OBS_MODES, TASKS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("env_id", TASKS)
def test_task_all_observations(env_id):
    subprocess.run([sys.executable, "-m", "tests.smoke", "env", "--env-id", env_id],
                   check=True, timeout=180)


@pytest.mark.parametrize("mode", OBS_MODES)
def test_training_and_evaluation(tmp_path, mode):
    subprocess.run([sys.executable, "-m", "tests.smoke", "train", "--obs-mode", mode,
                    "--output", str(tmp_path / mode)], check=True, timeout=300)
