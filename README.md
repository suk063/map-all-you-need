# ManiSkill PPO / BC benchmark

ManiSkill 3.0.1의 dense-reward table-top task를 위한 작은 PyTorch baseline입니다.
PPO와 BC는 같은 actor·관측 전처리·checkpoint 형식을 사용하고 `benchmark/eval.py` 하나로 평가합니다.

## 설치

Linux, Python 3.12, NVIDIA GPU와 Vulkan 드라이버 기준입니다. GPU 학습에는 CUDA PyTorch가 필요합니다.
CPU backend로 수집한 BC 데이터는 CPU 시뮬레이션에서 평가하지만, 영상 렌더링에는 Vulkan이 필요합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 이 저장소 안에 asset을 보관합니다. 이후 실행에서도 같은 값을 유지하세요.
export MS_ASSET_DIR="$PWD/.assets"
python -m mani_skill.utils.download_asset ycb
python -m mani_skill.utils.download_asset widowxai
```

`ycb`는 PickSingleYCB, `widowxai`는 PickCubeWidowXAI에 필요합니다.
나머지 지원 task의 asset은 패키지에 포함됩니다. 다운로드 명령의 기존 asset 삭제 질문에는
이미 설치되어 있다면 동의하지 않아도 됩니다.

## 코드 구조

```text
benchmark/common/envs.py      task 목록, 환경 생성, action 구성
benchmark/common/policy.py    관측 전처리, actor, save_policy / load_policy
benchmark/common/utils.py     seed와 결과 저장
benchmark/rl/train.py         PPO
benchmark/bc/train.py         MSE behavior cloning
benchmark/bc/data.py          ManiSkill HDF5/JSON lazy dataset
benchmark/eval.py             공통 평가
tests/                       단위 테스트와 실제 simulator smoke test
```

## 지원 task와 입력

[공식 table-top 목록](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html)의
dense reward가 있는 16개 task만 허용합니다. 목록은 버전과 함께 고정되며, 선택한 환경만 생성합니다.

| Task | 기본 episode 길이 | 공식 demo |
|---|---:|:---:|
| LiftPegUpright-v1 | 50 | O |
| PegInsertionSide-v1 | 100 | O |
| PickCube-v1 | 50 | O |
| PickCubeSO100-v1 | 50 | — |
| PickCubeWidowXAI-v1 | 50 | — |
| PickSingleYCB-v1 | 50 | — |
| PlaceSphere-v1 | 50 | — |
| PokeCube-v1 | 50 | O |
| PullCube-v1 | 50 | O |
| PullCubeTool-v1 | 100 | O |
| PushCube-v1 | 50 | O |
| PushT-v1 | 100 | O |
| RollBall-v1 | 80 | O |
| StackCube-v1 | 50 | O |
| TwoRobotPickCube-v1 | 100 | O |
| TwoRobotStackCube-v1 | 100 | O |

| `--obs-mode` | Policy 입력 |
|---|---|
| `state` | ManiSkill `obs_mode="state"`의 전체 벡터 |
| `rgb` | 모든 기본 관측 카메라의 RGB + 해당 모드의 `agent`, `extra` |
| `rgbd` | 모든 기본 관측 카메라의 RGB·depth + 해당 모드의 `agent`, `extra` |

ManiSkill 기본 관측을 그대로 사용합니다. 영상 모드에 원래 포함된 목표 좌표나 `is_grasped`도 유지하고,
별도로 `state` 모드를 요청해서 privileged 정보를 추가하지 않습니다.
state-only는 MLP(256, 256), 영상은 NatureCNN(256)과 state MLP(256, 256)의 출력을 연결합니다.
PPO의 critic은 actor encoder를 공유하며, BC에는 critic이 필요하지 않습니다.

영상은 카메라 이름 순서로 채널에 연결합니다. RGB는 bilinear, depth는 nearest로 128×128에 맞춥니다.
RGB는 `/255`, depth는 `/1000`으로 m 단위로 바꾸며 invalid depth 0은 유지합니다.
추가 state 항목 순서도 checkpoint에 저장합니다. 영상 augmentation이나 state 통계 정규화는 사용하지 않습니다.
관측 정의는 [ManiSkill 문서](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)를 참고하세요.

## PPO 학습

아래 명령은 저장소 루트에서 실행합니다.

```bash
python -m benchmark.rl.train --env-id PickCube-v1 --obs-mode state --seed 0
python -m benchmark.rl.train --env-id PickCube-v1 --obs-mode rgb --seed 0
python -m benchmark.rl.train --env-id TwoRobotPickCube-v1 --obs-mode rgbd --seed 0 --num-envs 16
```

기본값은 10M environment transitions, rollout 50 step, Adam learning rate 3e-4,
gamma 0.99, GAE lambda 0.95, PPO clip 0.2, 4 update epochs, minibatch 256입니다.
병렬 환경 수는 state 256개, 영상 32개입니다. 마지막 vector step 때문에 실제 transition 수는
요청보다 최대 `num_envs - 1`만큼 많을 수 있으며 실제 값을 로그에 기록합니다.

reward는 `dense`, controller는 `pd_joint_delta_pos`, robot과 episode 길이는 task 기본값입니다.
두 로봇에도 같은 controller를 적용하고 두 action을 하나의 벡터로 연결합니다.
학습은 성공 직후 reset하지 않고 전체 horizon을 사용합니다. task 기본 reconfiguration 설정을 유지하므로
YCB처럼 형상이 달라지는 환경도 전체 reset 때 객체를 다시 샘플링할 수 있습니다.
GAE는 시간 제한 시 마지막 관측의 value로 bootstrap하며 episode 사이에는 trace를 연결하지 않습니다.

```bash
# 짧은 실행 확인. 성능 평가용 학습량은 아닙니다.
python -m benchmark.rl.train --obs-mode rgbd --num-envs 2 --num-steps 50 \
  --total-timesteps 200 --update-epochs 1 --batch-size 32
python -m benchmark.rl.train --help
```

## BC 데이터와 학습

관측이 저장된 ManiSkill `.h5`와 같은 이름의 `.json`을 입력합니다.
`obs[t] -> actions[t]`로 학습하고 각 trajectory의 마지막 관측은 제외합니다.
영상은 sample별로 읽기 때문에 전체 이미지 데이터를 GPU/RAM에 올리지 않습니다.
`--num-demos`는 JSON에 기록된 순서의 앞 N개 trajectory를 사용합니다.
공식 RL demo의 action은 controller가 clip하기 전 값일 수 있습니다. BC는 저장된 action을
그대로 MSE 학습하고, policy 실행 시 controller 범위로 clip합니다.

공식 raw demo는 관측을 생략하는 경우가 있으므로 먼저 replay해야 합니다.
다음은 PickCube의 RL demonstration을 원래 controller/backend로 RGBD replay하는 예입니다.

```bash
python -m mani_skill.utils.download_demo PickCube-v1 -o demos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
  --obs-mode rgbd --use-env-states --save-traj --num-envs 1

python -m benchmark.bc.train \
  --demo-path demos/PickCube-v1/rl/trajectory.rgbd.pd_joint_delta_pos.physx_cuda.h5 \
  --obs-mode rgbd --epochs 100 --batch-size 256 --seed 0
```

state 또는 RGB는 replay와 학습의 `--obs-mode`를 각각 `state`, `rgb`로 바꿉니다.
replay를 시험할 때는 `--count 2`를 추가할 수 있습니다. task마다 원본 경로와 수집 방식이 다르므로
다운로드된 파일을 확인하세요. controller 변환은 공식 replay의 `--target-control-mode`를 사용합니다.
기존 데이터와 controller/backend가 맞는지 확인한 뒤 사용해야 합니다.
자세한 방법은 [replay 문서](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)에 있습니다.

BC는 JSON에서 task·robot·controller·backend·환경 kwargs를 복원합니다.
기본 설정이 metadata에서 생략되었다면 고정된 ManiSkill 버전의 기본값을 사용하며,
backend가 없는 구형 metadata는 공식 replay 규칙대로 CPU로 취급합니다.
데모는 관측 모드와 action 차원이 일치해야 합니다. 원본 데이터는 자동으로 변환하지 않습니다.
공식 demo가 없는 네 task는 사용자가 준비한 같은 형식의 데이터가 필요합니다.
두 로봇의 native action dictionary와 연결된 action 벡터를 모두 읽습니다.
ManiSkill 3.0.1 생성 API에 맞춰 두 로봇은 같은 controller를 사용해야 합니다.

## 공통 평가와 저장 결과

```bash
python -m benchmark.eval --checkpoint runs/rl/PickCube-v1/state/<run>/policy.pt
python -m benchmark.eval --checkpoint runs/bc/PickCube-v1/rgbd/<run>/policy.pt \
  --episodes 100 --seed 10000
```

필수 인자는 checkpoint 하나입니다. 모델 구조, 전처리, 환경, controller, backend는 자동 복원됩니다.
기본 100개 episode에서 deterministic action을 사용하고 정확히 요청한 episode 수만 집계합니다.
GPU는 기본 8개 병렬 환경, CPU는 1개 환경을 사용합니다.
성공 직후 종료하지 않고 저장된 horizon까지 실행하며, reset마다 `reconfiguration_freq=1`로 재구성합니다.
이 설정은 [ManiSkill의 평가 방식](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#evaluation)을 따릅니다.

학습 결과는 `runs/<rl|bc>/<task>/<mode>/seed<seed>-<timestamp>/`에 저장됩니다.
`--output`으로 새 디렉터리를 지정할 수 있습니다. 기존 학습 디렉터리는 덮어쓰지 않습니다.

- `config.json`: 학습 설정과 복원된 환경 설정.
- `train.csv`: PPO loss·학습 중 return/success 또는 BC MSE와 누적 시간.
- `policy.pt`: actor, 전처리 구성, action 범위, 환경, 버전, seed, 학습량. PPO critic·optimizer는 포함하지 않습니다.
- `eval-seed<seed>.json`: 설정 및 `success_once`, `success_at_end`, return, episode 길이의 평균.
- `eval-seed<seed>.csv`: episode별 평가 지표. 같은 출력 이름으로 재평가하면 평가 결과 파일을 갱신합니다.

policy를 Python에서 직접 불러올 수도 있습니다.

```python
from benchmark.common import load_policy
from benchmark.common.envs import make_env

policy = load_policy("runs/example/policy.pt", device="cuda")
env = make_env(policy.env_config, num_envs=1, evaluation=True)
try:
    obs, _ = env.reset(seed=10000)
    action = policy.act(obs)  # native batched ManiSkill observation -> flat batched action
    obs, reward, terminated, truncated, info = env.step(action.to(env.device))
finally:
    env.close()
```

개별 policy를 공정하게 비교하려면 task·입력·controller·backend·episode 길이·평가 seed와 병렬 환경 수를 맞추세요.
BC는 데이터 생성 방식과 demo 수, PPO는 환경 transition 수도 함께 기록하세요.
기본 hyperparameter는 출발점이며 모든 task에서 동일한 성능이나 수렴을 보장하는 설정은 아닙니다.
전체 task/seed 실행기, 자동 demo 수집, 학습 재개 기능은 포함하지 않습니다.

## 검증

```bash
python -m pip install pytest
python -m pytest -q                  # simulator 없이 데이터/GAE/checkpoint 검증
python -m pytest -m integration -q   # asset과 NVIDIA GPU 필요
```

통합 검사는 16 task × state/rgb/rgbd에서 reset·step, 유한한 dense reward와 action shape를 확인합니다.
PickCube에서는 각 입력에 대해 PPO·BC 짧은 학습 → 저장 → 공통 평가를 실행하며,
재로딩 전후 action 일치도 확인합니다. BC 통합 테스트의 짧은 zero-action trajectory는
데이터 경로 검사용이며 expert demonstration이나 학습 성능 측정 자료가 아닙니다.

구현 시 Python 3.12 / PyTorch 2.10.0+cu128 / RTX 4090에서 단위 테스트 16개와
통합 테스트 19개를 통과했습니다. 공식 PickCube RL demo 2개를 RGBD로 replay한 BC 학습·평가와
두 로봇 RGBD BC 학습·평가도 별도로 확인했습니다. 장시간 학습의 수렴·성공률은 측정하지 않았습니다.
