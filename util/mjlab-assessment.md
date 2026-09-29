# mjlab 적용 검토

현재는 Playground + JAX/Flax + Brax PPO를 유지하는 것이 적절합니다.
mjlab로 감싸는 것만으로 학습 속도나 성능이 좋아진다고 볼 근거는 없습니다.
아래는 소스 구조를 비교한 판단이며, 동일 task의 학습 속도·성공률 A/B 실험 결과가 아닙니다.

확인한 mjlab revision: `f135c1daa0f278bd19e323c2b9f256ca541ae2c5` (1.6.0).

| 항목 | 현재 코드와 비교 |
|---|---|
| 물리 속도 | 현재 10개 task 중 9개가 이미 MJX/Warp를 기본값으로 사용하고, Aero만 JAX입니다. mjlab도 MuJoCo Warp를 사용하므로 GPU 물리 가속을 새로 얻는 전환은 아닙니다. |
| 학습 경로 | 현재 JAX rollout·Flax encoder·Brax PPO가 연결되어 있습니다. mjlab은 PyTorch tensor API와 RSL-RL을 사용합니다. 단순 wrapper는 JAX/Torch 경계와 동기화 비용을 추가할 수 있습니다. DLPack으로 복사를 줄여도 두 실행 체계가 하나로 합쳐지는 것은 아닙니다. |
| 전체 이식 | mjlab의 manager 환경으로 action, reset, reward, termination, 관측 history·노이즈를 옮기고 map encoder와 checkpoint도 PyTorch/RSL-RL로 다시 연결해야 합니다. 공식 Playground task와 같은 동작인지 재검증해야 합니다. |
| 최종 결과 | 같은 reward·관측·학습량이라도 PPO 구현과 설정 변화로 결과가 달라질 수 있습니다. 프레임워크 변경 자체가 성공률을 높이지는 않습니다. |
| RGB | mjlab은 depth/raycast를 제공하지만 고품질 RGB는 공식 범위 밖입니다. 현재 Playground의 64×64 RGB + Brax CNN 경로를 그대로 대체하지 않습니다. |
| 개발 편의 | 많은 로봇에 재사용하는 manager 구조, actuator 설정, randomization, PyTorch 도구 사용에는 장점이 있습니다. 현재의 간결한 공식 task 재사용 목표에서는 이식 비용이 더 큽니다. |

특히 map 모드의 전체 point 처리와 Point Transformer 학습 비용은 환경 wrapper 교체만으로
없어지지 않습니다. 먼저 rollout과 PPO update 시간을 나누어 측정해야 합니다.
추후 비교한다면 같은 GPU, task, 관측, 환경 수, solver·control timestep, PPO 학습량을 고정하고
컴파일을 포함한 시작 시간, 준비 후 transitions/sec, update 시간, 최대 VRAM,
여러 seed의 return·task 지표를 각각 비교하는 것이 맞습니다.

설치도 현재 환경과 분리해야 합니다. 확인한 mjlab은 `mujoco~=3.11.0`,
`mujoco-warp~=3.11.0`, `torch>=2.14.0`, `rsl-rl-lib==5.5.1`을 요구합니다.
현재 고정한 MuJoCo/Warp 3.14.0, Torch 2.10.0과 충돌합니다.
mjlab은 설치하거나 training/eval 경로에 추가하지 않았습니다. mjviser는 독립적으로 사용합니다.

근거:

- [공식 설계·PyTorch API·RGB 범위](https://github.com/mujocolab/mjlab/blob/f135c1daa0f278bd19e323c2b9f256ca541ae2c5/docs/source/motivation.rst)
- [공식 의존성·RSL-RL 버전](https://github.com/mujocolab/mjlab/blob/f135c1daa0f278bd19e323c2b9f256ca541ae2c5/pyproject.toml)
- [MuJoCo Warp 실행과 CUDA graph](https://github.com/mujocolab/mjlab/blob/f135c1daa0f278bd19e323c2b9f256ca541ae2c5/src/mjlab/sim/sim.py)
