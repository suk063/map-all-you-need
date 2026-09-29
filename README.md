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
benchmark/common/dino.py      로컬 frozen DINOv3 로더
benchmark/common/mapping.py   multiview map cache, pose 기반 map 관측
benchmark/common/points.py    상대 위치 Point Transformer
benchmark/common/utils.py     seed와 결과 저장
benchmark/rl/train.py         PPO
benchmark/bc/train.py         MSE behavior cloning
benchmark/bc/data.py          ManiSkill HDF5/JSON lazy dataset
benchmark/bc/map_data.py      env_states 복원 및 map BC cache
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
| `rgb` | 128×128 RGB + 선택적 `agent`, `extra`; NatureCNN |
| `rgbd` | 128×128 RGB·depth + 선택적 `agent`, `extra`; NatureCNN |
| `dino` | 128×128 RGB + 선택적 `agent`, `extra`; frozen DINOv3 ViT-S+/16 |
| `map` | 3D 좌표 + DINOv3 ViT-L/16 특징; Point Transformer |

영상 policy는 기본적으로 state를 포함합니다. `--no-state-input`을 지정하면 `agent`, `extra`를
모두 제외하므로 `goal_pos`, `is_grasped` 등도 입력하지 않습니다. `--state-input`으로 명시적으로
포함할 수 있습니다. `state` 모드는 state가 필수이고, `map` 모드는 별도 state 입력을 허용하지 않습니다.

RL과 BC 모두 `--view external|wrist|all`로 policy에 넣을 카메라를 선택합니다. 기본값은 `all`입니다.

| `--view` | 선택되는 카메라 |
|---|---|
| `external` | wrist camera를 제외한 기존 관측 카메라 |
| `wrist` | 이름이 `hand_camera` 또는 `wrist_camera`로 끝나는 기존 카메라. 두 팔이면 양쪽 모두 사용 |
| `all` | 기존 관측 카메라 전체 |

카메라를 추가하거나 로봇을 교체하지 않고 policy 입력만 선택합니다. 환경은 기존 카메라를 계속 렌더링합니다.
선택한 종류의 카메라가 없으면 사용 가능한 이름과 함께 오류가 발생합니다.
예를 들어 기본 `PickCube-v1`에는 wrist camera가 없지만 `PegInsertionSide-v1`, `StackCube-v1`,
`PickSingleYCB-v1`, 두 `TwoRobot*` task에는 있습니다. `dino`에도 같은 카메라 선택을 적용합니다.
`state`, `map` 모드에서는 기본값 `all`만 허용하며, map의 사전 multiview 렌더링은 이 옵션과 무관합니다.
BC 데이터에도 선택한 카메라의 영상이 있어야 하며, 선택하지 않은 카메라 데이터는 읽지 않습니다.

State를 포함하는 영상 policy는 ManiSkill 기본 관측의 목표 좌표나 `is_grasped`를 유지하고,
별도로 `state` 모드를 요청해서 privileged 정보를 추가하지 않습니다.
state-only는 MLP(256, 256), 영상은 NatureCNN(256)과 state MLP(256, 256)의 출력을 연결합니다.
PPO의 critic은 actor encoder를 공유하며, BC에는 critic이 필요하지 않습니다.

영상은 카메라 이름 순서로 채널에 연결합니다. RGB는 bilinear, depth는 nearest로 128×128에 맞춥니다.
RGB는 `/255`, depth는 `/1000`으로 m 단위로 바꾸며 invalid depth 0은 유지합니다.
추가 state 항목 순서도 checkpoint에 저장합니다. 영상 augmentation이나 state 통계 정규화는 사용하지 않습니다.
관측 정의는 [ManiSkill 문서](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)를 참고하세요.

## Frozen DINO와 map policy

DINOv3 소스와 가중치는 로컬 파일을 사용하며 자동 다운로드하지 않습니다. 현재 머신에서는
인접한 `../reachy_task/external/dinov3`와 `../reachy_task/assets/dinov3`를 기본 경로로 사용합니다.
다른 위치에서는 `--dino-source`, `--dino-weights` 또는 `DINO_SOURCE`, `DINO_WEIGHTS`를 지정하세요.

- `dino`: `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`.
- `map`: `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.

이미지 DINO는 ImageNet 정규화 후 8×8×384 patch 특징을 추출합니다. 각 patch를 64차원으로
projection하고 공간 순서를 유지한 MLP로 256차원 출력을 만듭니다. Backbone은 항상 eval/frozen이며
PPO rollout에는 frozen 특징을 저장합니다. BC는 RGB demo의 필요한 sample만 읽습니다.
DINO는 RGB 전용이므로 공식 replay에는 `--obs-mode rgb`, 학습에는 `--obs-mode dino`를 사용합니다.

```bash
python -m benchmark.rl.train --obs-mode rgbd --no-state-input
python -m benchmark.rl.train --obs-mode dino --no-state-input
python -m benchmark.rl.train --env-id StackCube-v1 --obs-mode dino --view wrist
python -m benchmark.bc.train --demo-path demos/trajectory.rgb.h5 --obs-mode dino --no-state-input
```

Map policy는 [reachy-agent](https://github.com/suk063/reachy-agent)의 구성요소별 특징 추출과
[SERF](https://arxiv.org/abs/2606.12956)의 좌표 갱신 방식을 참고합니다. 이 벤치마크는 **시뮬레이터의
정확한 pose와 visual mesh를 사용**합니다. 영상 기반 object tracking이나 SERF의 latent-map 학습은
포함하지 않으므로 perception-only baseline과 구분해서 비교하세요.

별도 render-only scene에서 물체·기존 goal marker·robot link의 표면을 256×256으로 관측하여
frozen ViT-L/16의 1024차원 특징을 모읍니다. 기본 96개 구면 방향과 최대 512개 보충 시점을 사용하고,
관측된 특징을 평균합니다. 관측되지 않은 표면 점은 제거하며 coverage를 cache에 기록합니다.
Robot link를 분리해서 관측하므로 다른 link가 가리는 표면도 사전에 관측할 수 있습니다.
실제 robot을 움직이는 scan 동작은 필요하지 않습니다.

각 구성요소 local frame에서 **1.5cm voxel당 한 점**을 유지합니다. 서로 다른 물체·link는 합치지 않으며
정해진 point 수로 입력을 자르지 않습니다. 이후에는 특징을 고정하고 simulator pose로 좌표만 갱신합니다.
Reset에서 형상·재질이 달라지면 해당 구성요소만 새로 준비합니다. 기본 cache는 `.cache/maps`이며
`--map-cache`로 변경할 수 있습니다. 가중치 hash, DINO 소스 revision, 형상·재질과 추출 설정을 cache key에 반영합니다.

| 옵션 | 기본값 | 동작 |
|---|---|---|
| `--map-robot full\|gripper` | `full` | 전체 robot 또는 손·손가락만 포함. 양팔은 양쪽 모두, panda_stick은 말단 도구 포함 |
| `--map-background table\|none` | `none` | tabletop 윗면 포함 여부. 환경의 table/collision은 유지 |
| `--map-views` | `96` | 구성요소별 기본 시점 수 |
| `--map-extra-views` | `512` | 미관측 표면을 위한 최대 보충 시점 수 |

기존 goal marker는 시각 요소로 map에 포함하며 collision을 추가하지 않습니다.
기존 marker가 없는 task에는 새 marker를 만들지 않습니다. Map 관측에는 image·qpos·goal_pos·grasp flag를
별도 입력하지 않습니다. Actor와 critic 모두 같은 map encoder를 사용합니다.

[SERF-VLA의 Point Transformer](https://github.com/ExistentialRobotics/SERF-VLA/blob/ea27b7aa753cf7da6def975846ccb5d3180e46f7/src/serf_b1k/models/point_transformer_local.py)를
작게 옮겼습니다. DINO projection 1024→64, 폭 64/128의 두 stage, 이웃 16개, 내부 대표점 최대 256→64개와
마지막 전체 대표점 attention을 사용합니다. **전체 map에서 256차원 token 하나**를 attention pooling으로
뽑습니다. Encoder는 342,457개 학습 파라미터입니다. 절대 xyz를 feature에 붙이지 않고 상대 위치만 인코딩합니다.
Sampling은 평면의 거리 동률을 안정화하는 0.1mm 상대 좌표 grid를 사용하고, attention은 원래 m 단위 offset을 사용합니다.

```bash
python -m benchmark.rl.train --obs-mode map --map-robot gripper --map-background none
python -m benchmark.rl.train --obs-mode map --map-robot full --map-background table
python -m benchmark.bc.train --obs-mode map --demo-path demos/trajectory.h5
```

Map BC는 관측 이미지 대신 **T+1개의 `env_states`**, T개의 actions, episode seed와 metadata가 필요합니다.
각 `env_states[t]`를 원래 backend에서 복원하여 `actions[t]`에 대응하는 map을 캐시한 뒤 shuffled batch로
학습합니다. `obs`가 없는 공식 raw demo도 `env_states`가 있으면 사용할 수 있습니다. BC cache도 `.cache/maps` 아래에 저장합니다.
PPO는 map 좌표와 고정 특징의 ID만 rollout에 저장하고, 가변 point 수는 padding mask로 처리합니다.

## PPO 학습

아래 명령은 저장소 루트에서 실행합니다.

```bash
python -m benchmark.rl.train --env-id PickCube-v1 --obs-mode state --seed 0
python -m benchmark.rl.train --env-id PickCube-v1 --obs-mode rgb --seed 0
python -m benchmark.rl.train --env-id TwoRobotPickCube-v1 --obs-mode rgbd --seed 0 --num-envs 16
python -m benchmark.rl.train --env-id PegInsertionSide-v1 --obs-mode rgbd --view wrist --seed 0
python -m benchmark.rl.train --env-id StackCube-v1 --obs-mode rgb --view external --seed 0
```

기본값은 10M environment transitions, rollout 50 step, Adam learning rate 3e-4,
gamma 0.99, GAE lambda 0.95, PPO clip 0.2, 4 update epochs, minibatch 256입니다.
병렬 환경 수는 state 256개, 영상 32개, map 8개입니다. Map minibatch는 기본 16입니다.
마지막 vector step 때문에 실제 transition 수는
요청보다 최대 `num_envs - 1`만큼 많을 수 있으며 실제 값을 로그에 기록합니다.

reward는 `dense`, controller 기본값은 모든 입력 모드에서 `pd_ee_delta_pose`입니다.
RL·BC 모두 `--control-mode`로 controller를 선택합니다. 예를 들어
`--control-mode pd_joint_delta_pos` 또는 `--control-mode pd_ee_delta_pos`를 사용할 수 있습니다.
Panda의 `pd_ee_delta_pose` action은 위치 변화량 3개, 회전 변화량 3개, gripper 1개입니다.
Robot과 episode 길이는 task 기본값입니다.
ManiSkill 3.0.1의 SO100·WidowXAI는 EE controller를 제공하지 않으므로,
`PickCubeSO100-v1`·`PickCubeWidowXAI-v1`에서는 `--control-mode pd_joint_delta_pos`를 지정하세요.
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
`--num-demos`를 생략하면 전체 데모를 사용하고, 지정하면 JSON 순서의 앞 N개 trajectory를 사용합니다.
공식 RL demo의 action은 controller가 clip하기 전 값일 수 있습니다. BC는 저장된 action을
그대로 MSE 학습하고, policy 실행 시 controller 범위로 clip합니다.

공식 raw demo는 관측을 생략하는 경우가 있으므로 먼저 replay해야 합니다.
다음은 PickCube의 RL demonstration을 원래 controller/backend로 RGBD replay하는 예입니다.

```bash
python -m mani_skill.utils.download_demo PickCube-v1 -o demos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
  --obs-mode rgbd --use-env-states --save-traj --num-envs 1

python -m benchmark.bc.train \
  --demo-path demos/PickCube-v1/rl/trajectory.rgbd.pd_ee_delta_pose.physx_cuda.h5 \
  --obs-mode rgbd --epochs 100 --batch-size 256 --seed 0
```

BC에서도 `--view wrist` 또는 `--view external`을 추가하면 같은 데이터에서 해당 카메라만 사용합니다.
위 PickCube 예제에는 wrist 영상이 없으므로 `all` 또는 `external`을 사용합니다.

state 또는 RGB는 replay와 학습의 `--obs-mode`를 각각 `state`, `rgb`로 바꿉니다.
replay를 시험할 때는 `--count 2`를 추가할 수 있습니다. task마다 원본 경로와 수집 방식이 다르므로
다운로드된 파일을 확인하세요. controller 변환은 공식 replay의 `--target-control-mode`를 사용합니다.
기존 데이터와 controller/backend가 맞는지 확인한 뒤 사용해야 합니다.
자세한 방법은 [replay 문서](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)에 있습니다.

BC는 JSON에서 task·robot·controller·backend·환경 kwargs를 복원합니다.
선택한 `--control-mode`(기본 `pd_ee_delta_pose`)가 데모의 controller와 다르면 학습 전에 오류를 냅니다.
다른 controller로 수집한 데모는 동일한 `--control-mode`를 지정하거나 공식 replay로 변환해야 합니다.
Eval은 CLI 기본값과 관계없이 checkpoint에 저장된 controller를 복원합니다.
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
State·robot·background 선택도 자동 복원합니다. DINO 이미지 checkpoint에는 frozen backbone 가중치도 들어갑니다.
Map checkpoint는 사용한 특징 cache 파일을 참조하므로 이 파일들을 보관하세요. 다른 머신에서는 기록된 DINO 소스·map cache
경로를 사용할 수 있어야 하며, 새 형상 map 생성에는 ViT-L 가중치도 필요합니다. 이 외부 파일들은 Git에서 제외됩니다.
학습에 사용한 view와 카메라 이름도 checkpoint에서 복원하므로 eval에는 `--view`를 별도로 전달하지 않습니다.
view 옵션 추가 전의 checkpoint도 기존처럼 모든 카메라를 사용하는 `all`로 불러옵니다.
기본 100개 episode에서 deterministic action을 사용하고 정확히 요청한 episode 수만 집계합니다.
GPU는 기본 8개 병렬 환경, CPU는 1개 환경을 사용합니다.
성공 직후 종료하지 않고 저장된 horizon까지 실행하며, reset마다 `reconfiguration_freq=1`로 재구성합니다.
이 설정은 [ManiSkill의 평가 방식](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#evaluation)을 따릅니다.

학습 결과는 `runs/<rl|bc>/<task>/<mode>/seed<seed>-<timestamp>/`에 저장됩니다.
`--output`으로 새 디렉터리를 지정할 수 있습니다. 기존 학습 디렉터리는 덮어쓰지 않습니다.

- `config.json`: 학습·관측·환경 설정, 준비 시간, 학습 파라미터 수. Map PPO는 초기 point 수도 기록합니다.
- `train.csv`: PPO loss·return/success 또는 BC MSE, 누적 시간, samples/s와 PyTorch peak GPU 할당량(MiB).
- `policy.pt`: actor, 전처리 구성, action 범위, 환경, 버전, seed, 학습량. PPO critic·optimizer는 포함하지 않습니다.
- `eval-seed<seed>.json`: 설정 및 `success_once`, `success_at_end`, return, episode 길이의 평균.
- `eval-seed<seed>.csv`: episode별 평가 지표. 같은 출력 이름으로 재평가하면 평가 결과 파일을 갱신합니다.

policy를 Python에서 직접 불러올 수도 있습니다.

```python
from benchmark.common import load_policy
from benchmark.common.envs import make_env

policy = load_policy("runs/example/policy.pt", device="cuda")
env = make_env(policy.env_config, num_envs=1, evaluation=True, map_bank=policy.map_bank)
try:
    obs, _ = env.reset(seed=10000)
    action = policy.act(obs)  # batched observation -> flat batched action
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

초기 구현은 Python 3.12 / PyTorch 2.10.0+cu128 / RTX 4090에서 단위 테스트 16개와
통합 테스트 19개를 통과했습니다. 공식 PickCube RL demo 2개를 RGBD로 replay한 BC 학습·평가와
두 로봇 RGBD BC 학습·평가도 별도로 확인했습니다. 장시간 학습의 수렴·성공률은 측정하지 않았습니다.

view 선택은 카메라 필터링, BC의 선택된 영상만 읽기, 기존 checkpoint 호환성을 단위 테스트로 검사합니다.
추가 통합 테스트는 PegInsertionSide의 RGB `external`과 TwoRobotPickCube의 RGBD `wrist`로
RL·BC 학습부터 checkpoint 기반 평가까지 확인합니다.

DINO/map 추가 구현은 같은 RTX 4090 환경에서 **단위 테스트 40개, 통합 테스트 26개**를 나누어 실행해 통과했습니다.

- 16개 task: gripper-only/background 없음, 실제 ViT-L 특징으로 reset·step·역전파·자동 reset을 확인했습니다.
- PickCube: DINO의 state 포함/제외, map의 full+table/gripper-only, RGB·RGBD의 state 제외에 대해
  PPO·BC 학습 → 저장 → 공통 eval → 재로딩 action 일치를 확인했습니다. 기존 state/RGB/RGBD 학습도 재검사했습니다.
- Map BC: 온라인 map과 `env_states[t]` 복원 결과, `actions[t]` 정렬, cache의 feature ID 재매핑을 검사했습니다.
  관측 없는 공식 PickCube RL raw demo 2개로도 map BC 학습과 GPU 평가를 실행했습니다.
- 통합 테스트는 시간을 줄이기 위해 8개 시점·보충 시점 0개를 사용합니다. **기본 96+512 설정도 별도 검증**했습니다.
  PickCube full+table은 환경당 17,635점, 두 환경의 첫 map 준비 약 80.5초였고 이후 cache를 재사용했습니다.
  Gripper-only/background 없음의 기본 설정으로 PPO 104 transition과 공통 eval도 실행했습니다.

마지막 짧은 map PPO 실행(환경 2개, batch 8, update epoch 1)은 약 44.4 transition/s, PyTorch peak 약 270MiB였습니다.
이는 초기 준비 시간을 제외한 smoke 실행 수치이며 Vulkan 메모리와 장시간 학습 성능을 나타내지 않습니다.
실행 기록은 로컬 `runs/map-default-settings.json`, `runs/map-default-ppo/`, `runs/official-map-bc/`에 있습니다.
전체 task의 기본 96+512 coverage, 장시간 수렴·성공률, 다른 GPU에서의 처리량은 검증하지 않았습니다.

`pd_ee_delta_pose` 기본값과 BC controller 선택 추가 후 단위 테스트 41개,
학습·평가 통합 테스트 10개, 16개 task의 state/RGB/RGBD reset·step 검사가 통과했습니다.
SO100·WidowXAI 검사는 지원되는 `pd_joint_delta_pos`를 명시적으로 사용합니다.
