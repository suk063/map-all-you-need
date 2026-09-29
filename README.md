# MuJoCo Playground observation benchmark

MuJoCo Playground manipulation task에서 `state`, `rgb`, `map` 입력을 비교하는 작은 JAX/Brax PPO benchmark입니다.
환경과 PPO는 공식 구현을 사용하고, map 생성과 Point Transformer만 별도로 구현합니다.

## 설치

Linux, Python 3.12, NVIDIA GPU 기준입니다. MuJoCo EGL 렌더링을 위한 NVIDIA 드라이버가 필요합니다.
기존 환경과 의존성이 다르므로 새 가상환경에서 설치하는 것을 권장합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Playground `ef4fefc13033c0468af4ef651847f5348af0c7d7`, Brax
`8253c01f208a0ce2e008bf214fb4e8cd67efd609` revision을 고정합니다.
JAX CUDA 12, MuJoCo/MJX 3.14.0을 사용합니다. 첫 환경 생성 시 Playground가
고정된 MuJoCo Menagerie asset을 자동 다운로드하고, 최초 실행 시 JAX/Warp가 컴파일합니다.

DINO 추론은 `map`의 특징 캐시 생성 때만 실행합니다. 소스·가중치는 자동 다운로드하지 않습니다.
기본 경로는 인접한 `../reachy_task/external/dinov3`와
`../reachy_task/assets/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`입니다.
다른 위치에서는 `--dino-source`, `--dino-weights` 또는 `DINO_SOURCE`, `DINO_WEIGHTS`를 지정하세요.
새 map 학습 실행은 캐시를 식별하기 위해 로컬 가중치의 hash를 확인합니다.
PyTorch는 사전 특징 추출에만 사용하며 policy와 PPO 학습은 모두 JAX입니다.

## Task와 입력

기본 task는 `PandaPickCubeCartesian`입니다. Non-Prehensile Manipulation을 포함하여
공식 `manipulation.ALL_ENVS` 전체를 지원합니다. 고정한 revision의 목록은 다음 10개입니다.

| Task | Action 차원 | 지원 입력 | 별도 goal marker |
|---|---:|---|---|
| AlohaHandOver | 14 | state, rgb, map | 표시 |
| AlohaSinglePegInsertion | 14 | state, rgb, map | 없음: peg와 socket의 삽입 관계 |
| PandaPickCube | 8 | state, rgb, map | 표시 |
| PandaPickCubeOrientation | 8 | state, rgb, map | 표시 |
| PandaPickCubeCartesian | 3 | state, rgb, map | 표시 (RGB 포함) |
| PandaOpenCabinet | 8 | state, rgb, map | 표시 |
| PandaRobotiqPushCube | 7 | state, rgb, map | 표시 |
| LeapCubeReorient | 16 | state, rgb, map | 표시 |
| LeapCubeRotateZAxis | 16 | state, rgb, map | 없음: z축 지속 회전 |
| AeroCubeRotateZAxis | 7 | state, rgb, map | 없음: z축 지속 회전 |

Action은 task에 직접 전달합니다. 제어 방식·스케일·reward·episode 길이·종료 조건을 바꾸지 않습니다.
예를 들어 `PandaPickCube`는 관절 위치 변화량, `PandaPickCubeCartesian`은 해당 task의
3차원 Cartesian action, `PandaRobotiqPushCube`는 해당 task의 torque action을 사용합니다.
성공이나 실패로 task가 종료되면 공식 wrapper가 reset합니다.
목표 위치·자세가 task를 정의하는 경우 goal marker를 표시합니다. 이 규칙은 state 환경의 시각화,
RGB 영상, map에 공통으로 적용하며 marker의 원본 형상·재질을 사용합니다.
`AlohaSinglePegInsertion`에는 별도 marker를 추가하지 않고 조작 대상인 peg와 socket을 유지합니다.
두 `RotateZAxis` task의 사용하지 않는 goal 형상은 state 시각화에서도 숨기며 map에도 넣지 않습니다.
State의 수치 관측·목표 pose·reward·성공 판정·collision은 공식 설정을 유지합니다.

`PandaPickCubeCartesian` RGB는 원본과 달리 goal marker를 목표 위치에 렌더링합니다.
렌더링할 때만 marker 위치를 바꾼 데이터 사본을 사용하므로 물리 상태와 RNG 진행은 변경하지 않습니다.
정책에는 marker가 포함된 RGB 영상만 전달하며 별도 goal 좌표나 state 벡터를 추가하지 않습니다.

- **state**: 공식 관측과 MLP를 사용합니다. 손 task의 history·노이즈, 공식 privileged-state critic도 유지합니다.
- **rgb**: native state 환경에 공통 batch renderer를 적용하여 `pixels/view_0`, 64×64 RGB와 Brax CNN을 사용합니다.
  원본 모델의 texture를 활성화하고 task별 goal 표시 규칙을 적용합니다.
  공식 action repeat·autoreset 이후 렌더링하고 추가 state 입력을 하지 않습니다.
  Actor와 critic은 각각 CNN을 가지며 둘 다 RGB만 받습니다.
- **map**: 기본 비영상 환경의 관측을 아래의 component map으로 교체합니다.
  Actor와 critic은 각각 Point Transformer를 가지며 둘 다 map만 받습니다.

신규 RGB는 모든 task에서 state/map과 같은 물리·reward·성공·종료 조건을 사용합니다.
Cartesian도 native state 설정을 사용하며, 기존 format-2 Cartesian RGB checkpoint는
저장된 공식 vision 설정으로 계속 평가합니다. RGB renderer는 MJX/Warp를 사용하고,
영상 생성 전용 모델을 분리하여 state/map의 simulator 설정을 유지합니다.
카메라는 Aloha `overhead_cam`, Cartesian `front`, Cabinet·Leap·Aero `side`이며,
카메라가 없는 Panda Pick 계열과 PushCube에는 scene 기본 시점의 고정 카메라를 추가합니다.
[공식 환경 소스](https://github.com/google-deepmind/mujoco_playground/blob/ef4fefc13033c0468af4ef651847f5348af0c7d7/mujoco_playground/_src/manipulation/franka_emika_panda/pick_cartesian.py),
[공식 PPO 설정](https://github.com/google-deepmind/mujoco_playground/blob/ef4fefc13033c0468af4ef651847f5348af0c7d7/mujoco_playground/config/manipulation_params.py).

## Map

Map은 **정확한 simulator pose와 visual geometry를 사용하는 관측**입니다. 영상 기반 tracking은 아닙니다.
로봇 link·물체를 별도의 MuJoCo render model에서 분리해 256×256으로 관측합니다.
형상·재질·texture를 유지하고 고정 조명을 사용합니다. 캐시 생성은 task의 모델·물리 상태를 수정하지 않습니다.

구성요소별 local frame에서 1.5cm voxel당 한 점을 유지하고, frozen DINOv3 ViT-L/16의
1024차원 특징을 multiview 평균합니다. 기본 96개 구면 시점과 최대 512개 보충 시점을 사용하며,
관측되지 않은 점은 제외합니다. 점 수 제한으로 전체 map을 자르지는 않습니다.
그다음에는 특징을 고정하고 MuJoCo의 forward kinematics로 계산된 body pose로 좌표와 normal을 갱신합니다.
관절이 있는 로봇과 조작 대상은 body별 local map을 유지하므로 각 part가 독립적으로 움직입니다.
한 body에 고정된 여러 geom은 같은 part로 취급합니다. Goal marker도 별도의 component로 포함합니다.
원본이 사용하는 투명도·색상·texture를 유지하고, 현재 mocap 위치·회전을 읽어 map을 갱신합니다.
Task가 물리 step 이후 goal을 갱신해도 바로 반영하며 simulator 상태나 RNG를 변경하지 않습니다.
`--map-robot full`은 로봇 전체 visual part를, `gripper`는 조작부만 포함합니다.
Panda는 hand와 양쪽 손가락, Robotiq은 gripper base와 모든 finger linkage,
Aloha는 양쪽 gripper, Leap/Aero는 손바닥과 모든 손가락을 포함합니다.
조작 대상의 articulated part와 task 관련 scene object는 두 설정 모두 유지합니다.
제외한 팔·고정 mount는 `--map-background true`로도 다시 포함하지 않습니다.
선택값은 실행 설정에 저장하며 checkpoint 평가에서도 자동으로 복원합니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--map-robot full\|gripper` | `full` | 로봇 전체 또는 물체 조작에 참여하는 손·gripper |
| `--map-background true\|false` | `false` | Task와 무관한 scene object도 포함. 바닥·벽은 항상 제외 |
| `--map-cache` | `.cache/maps` | 구성요소별 HDF5 cache |
| `--map-views` | `96` | 기본 시점 수 |
| `--map-extra-views` | `512` | 최대 보충 시점 수 |

바닥·벽·barrier와 모든 plane, collision proxy, site/tendon 보조 표시는 map에 넣지 않습니다.
환경의 collision과 tendon 물리는 그대로 유지합니다.
Task에 필요한 goal은 `map-background`와 `map-robot` 설정에 관계없이 포함합니다.

로봇·조작 대상·goal 이외의 scene 구성은 다음과 같습니다.

| Task | 추가 scene object |
|---|---|
| AlohaHandOver, AlohaSinglePegInsertion | 테이블·다리, 지지 프레임, 카메라 하우징, 바닥. HandOver에는 threshold 표시 plane도 있음 |
| PandaPickCube, PandaPickCubeOrientation | 바닥 |
| PandaPickCubeCartesian | 바닥, 초기 영역 표시 plane |
| PandaOpenCabinet | 바닥, 고정 barrier plane |
| PandaRobotiqPushCube | 바닥, 벽, pad, 투명 camera tracking box |
| LeapCubeReorient, LeapCubeRotateZAxis, AeroCubeRotateZAxis | 바닥, 손 고정 mount |

Aloha tabletop은 두 task의 `no_table_collision` reward에 사용되므로 항상 윗면을 포함합니다.
테이블 다리·지지 프레임·카메라 하우징, PushCube의 pad·camera tracking box는
`--map-background true`일 때 추가합니다. World에 직접 붙은 배경 geom도 각각 분리해 캐시합니다.
Cabinet barrier와 PushCube wall은 reward/종료 판정에 관련되지만 바닥·벽 필터를 우선합니다.
완전히 투명한 형상은 포함하지 않습니다. 기존 `table|none` map 설정은 지원하지 않습니다.

Cache key에는 MuJoCo 버전, 형상·재질·texture, 추출 설정, DINO 가중치 hash와 소스 revision을 반영합니다.
캐시에는 coverage와 body-local 단위 surface normal도 기록합니다. 현재 cache version은 3이며,
normal이 없는 이전 캐시는 덮어쓰지 않고 새 key로 생성합니다.

Point와 normal은 하나의 robot frame으로 변환합니다. Panda는 `link0`, Leap은 `leap_mount`,
Aero는 `tetheria_mount`가 기준입니다. Aloha는 양팔 base 위치의 중점을 원점으로 하고
`left/base_link`의 축을 사용합니다. Robot·물체·goal·배경 모두 같은 frame을 사용합니다.
매 reset에서 전체 유효 point의 min/max로 중심 `c=(min+max)/2`와
단일 스케일 `s=max(max(max-min)/2, 1e-6 m)`를 구하고 `xyz=(p_robot-c)/s`로 변환합니다.
세 축에 같은 스케일을 적용해 방향·각도·거리 비율을 보존합니다. 중심과 스케일은 episode 동안
고정하고 cached autoreset에서도 초기 관측과 일치하게 유지합니다. `[-1,1]`은 reset에서만
보장하며, 이후 범위를 벗어나도 clipping하지 않습니다. Normal은 robot-frame 단위벡터입니다.

Point Transformer는 기존 SERF-VLA 기반 구조를 Flax로 이식했습니다.
DINO projection 1024→64, 두 stage(폭 64/128, 대표점 최대 256/64), 이웃 16개,
마지막 전체 대표점 attention과 256차원 token 하나를 사용합니다.
Actor·critic 가중치는 분리합니다. Positional MLP는 7D edge feature
`[d, ||d||, ni·nj, ni·unit(d), nj·unit(d)]`를 받고 attention과 value/message 양쪽에 사용합니다.
여기서 `d=xyz_j-xyz_i`이며 절대좌표·body ID·정규화 중심·스케일을 학습 feature로 넣지 않습니다.
고정 shape와 mask로 padding을 처리하며 rollout에는 `xyz`, `normals`, `feature_ids`를 저장합니다.
보조 관측 `geometry_epsilon=1e-6 m / s`는 영거리·근접 pair의 방향을 안전하게 0으로 만드는
수치 처리에만 사용하며 MLP feature에는 포함하지 않습니다. Task의 native 관측 history는 내부에 보존합니다.

## 학습

```bash
python -m benchmark.rl.train --obs-mode state
python -m benchmark.rl.train --obs-mode rgb
python -m benchmark.rl.train --obs-mode map
python -m benchmark.rl.train --obs-mode map --map-robot gripper
python -m benchmark.rl.train --env-id AlohaHandOver --obs-mode map --map-background true
python -m benchmark.rl.train --env-id PandaRobotiqPushCube --obs-mode state
python -m benchmark.rl.train --env-id PandaRobotiqPushCube --obs-mode map
python -m benchmark.rl.train --help
```

State는 공식 task별 PPO hyperparameter와 학습량을 기본값으로 사용합니다.
RGB는 동일한 state 학습량과 PPO 설정에 CNN을 결합하고 환경 수 128, 평가 환경 수 8,
`batch_size=16`, 관측 정규화 비활성화를 적용합니다.
Map도 해당 task의 state PPO 설정을 사용하되 환경 수 8, 평가 환경 수 8,
Brax `batch_size=1`, PPO 관측 정규화 비활성화를 적용합니다. 위의 episode별 기하 정규화 외에
좌표·normal·특징 ID에 running normalization은 적용하지 않습니다.
Domain randomization은 공식 학습 script의 기본값처럼 별도로 활성화하지 않습니다.
Task 자체의 초기 상태·관측 노이즈·perturbation 설정은 유지합니다.

`--num-envs`, `--num-eval-envs`, `--total-timesteps`, `--unroll-length`, `--batch-size`,
`--num-minibatches`, `--num-updates-per-batch`, `--learning-rate` 등으로 명시적으로 변경할 수 있습니다.
Brax의 `batch_size`는 unroll sequence 수이며 `batch_size * num_minibatches`는 `num_envs`로 나누어져야 합니다.
학습은 Brax의 batch 단위로 진행하므로 실제 transition 수가 요청보다 많을 수 있습니다.
기록된 실제 step 수를 사용하세요. 기본 simulator 구현은 task 설정을 유지하며,
해당 task가 지원하면 `--impl jax|warp`로 변경할 수 있습니다.

```bash
# 세 모드 모두 사용할 수 있는 짧은 실행. 성능 평가용 학습량은 아닙니다.
python -m benchmark.rl.train --obs-mode state \
  --num-envs 2 --num-eval-envs 1 --total-timesteps 8 \
  --unroll-length 2 --batch-size 2 --num-minibatches 1 \
  --num-updates-per-batch 1 --num-evals 1 --no-run-evals
```

## 저장과 평가

기본 출력은 `runs/rl/<task>/<mode>/seed<seed>-<timestamp>/`입니다.
`--output`으로 새 디렉터리를 지정할 수 있으며 기존 디렉터리는 덮어쓰지 않습니다.

- `config.json`: 실제 환경·관측 shape·PPO 설정, seed, 버전, map cache 참조.
- `train.csv`: `steps,metric,value` 형식의 공식 Brax 학습·평가 지표.
- `checkpoints/<step>/`: 공식 Brax/Orbax checkpoint.
- `eval-seed<seed>.json`, `.csv`: 평가 요약 및 episode별 결과.

```bash
python -m benchmark.eval --checkpoint runs/rl/PandaPickCubeCartesian/state/<run>
python -m benchmark.eval --checkpoint runs/rl/PandaPickCubeCartesian/map/<run> \
  --episodes 100 --num-envs 8 --seed 10000
```

평가에는 실행 디렉터리를 전달하며 가장 마지막 checkpoint를 복원합니다.
기본 100개 episode를 deterministic action으로 평가하고 정확히 요청한 수만 집계합니다.
각 병렬 batch는 새로운 seed로 reset하고, 각 환경의 첫 완료 episode만 집계합니다.
Return·episode 길이와 task 지표의 누적값(`sum/`)·종료 시 값(`final/`)을 저장합니다.
성공 지표가 없는 task에는 임의의 성공 기준을 추가하지 않습니다.

`--resume <이전 실행 디렉터리> --output <새 디렉터리>`로 저장된 actor·critic·normalizer를
복구하고 전체 목표에서 저장된 누적 step을 뺀 만큼 학습합니다. Optimizer와 RNG는 초기화됩니다.
읽을 수 없는 불완전 checkpoint는 제외합니다. `--checkpoint-steps 1000000`으로
중간 평가 여부와 독립적으로 checkpoint 저장 간격을 지정할 수 있습니다.

Map 실행 디렉터리가 참조하는 cache 파일도 보관해야 합니다. 실행 디렉터리의 `map-cache/`에
해당 파일을 동봉하면 옮긴 위치를 우선 사용합니다. cache가 있으면 DINO 가중치를 다시 읽지 않습니다.
Goal 표시 규칙은 `config.json`에 `goal_markers: "task"`로 기록합니다. 이전 bool 설정의 실행은
관측 조건을 섞지 않도록 당시 소스로 평가하거나 현재 설정으로 새로 학습해야 합니다.
단, 기존 format-2 Cartesian RGB는 저장된 vision·goal 설정으로 평가하는 호환 경로를 유지합니다.
기존 실행·cache 파일은 보존하며 변경 없는 component cache는 재사용합니다.
새 map 실행은 `map_geometry` metadata에 `contact_robot_v1`, frame과 정규화 규칙을 기록합니다.
이 metadata가 없거나 다른 기존 map checkpoint의 평가·resume은 재학습 안내 오류로 거부합니다.
기존 state/RGB 호환성은 유지합니다.
기존 PyTorch `.pt` checkpoint, BC, `rgbd`, 독립 `dino` 모드와
`--control-mode`, `--view`, `--state-input` 옵션은 지원하지 않습니다.

## 코드와 검증

Mac에서 30개 cluster Job을 제출·agent 감시·복구·다운로드하는 실행법은
[cluster/README.md](cluster/README.md)에 정리했습니다. 본 학습 전에 같은 이미지의 GPU smoke test를 통과해야 합니다.

`util/view_scene.py`는 [mjviser](https://github.com/mujocolab/mjviser)의 브라우저 viewer로
10개 task를 모두 지원합니다. 공식 task의 seed별 reset 상태를 복사해 정지 상태로 시작합니다.

```bash
python -m pip install -r util/requirements.txt
python -m util.view_scene --env-id PandaOpenCabinet --port 8080
python -m util.view_scene --env-id AlohaHandOver --map-only --map-background true
python -m util.view_scene --env-id PandaPickCubeCartesian --map-only --map-robot gripper
python -m util.view_scene --help
```

출력된 `http://127.0.0.1:8080`을 열어 관절·actuator slider, camera, 재생/정지 기능을 사용할 수 있습니다.
`--map-only`는 map에 들어갈 visual geometry를 보여주며 DINO cache를 만들지 않습니다.
일반 scene 보기에는 바닥·벽도 보입니다. 두 보기 모두 위 표의 task별 goal 표시 규칙을 따릅니다.
재생은 CPU MuJoCo에서 현재 actuator control을 유지하는 scene 검사이며, Playground의 action/reward loop나
학습 policy 평가가 아닙니다. Reset은 최초의 task reset 상태를 복원합니다.
원격 머신에서는 `ssh -L 8080:127.0.0.1:8080 <host>`로 접속할 수 있습니다.

mjlab 적용 검토는 [util/mjlab-assessment.md](util/mjlab-assessment.md)에 정리했습니다.

`benchmark/common/envs.py`는 환경 설정, `policy.py`는 공식 network factory와 checkpoint,
`mapping.py`는 MuJoCo map, `points.py`는 Flax encoder를 담당합니다.
`benchmark/rl/train.py`와 `benchmark/eval.py`가 학습·평가 진입점입니다.

```bash
python -m pip install pytest
python -m pytest -q
# 실제 GPU, Menagerie asset, 로컬 DINO, util/requirements.txt가 필요한 검사
python -m pytest -q -m integration
```

통합 검사는 manipulation 전체의 native/map 물리 결과와 autoreset을 비교하고,
세 모드의 짧은 PPO 학습·파라미터 갱신·checkpoint 복원·평가를 확인합니다.
관절을 움직였을 때 body별 map과 geom의 좌표가 일치하는지 검사하며,
필요한 goal의 원본 형상·재질 보존, 불필요한 marker의 비표시와 map의 mocap pose 갱신을 확인합니다.
RGB에서는 marker 표시·이동에 따른 영상 변화와 원본 대비 물리 상태·reward·종료·RNG 보존을 검사합니다.
MJX/Warp의 접촉 계산은 동일한 원본 환경의 반복 실행에서도 작은 수치 차이를 보이므로,
물리 비교에는 관측한 GPU 오차 범위의 허용치를 사용합니다. 종료 결과와 RNG는 정확히 비교합니다.
Map smoke test는 실행 시간을 줄이기 위해 4개 시점과 보충 시점 0개를 사용합니다.
장기 학습 성능은 별도로 평가해야 합니다.
