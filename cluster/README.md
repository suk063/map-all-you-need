# Mac → Nautilus RL

기본값은 **10 task × state/rgb/map × seed 0 = 30개 독립 Job**이다.
각 policy에 task의 공식 state PPO 학습량을 사용하고, 최대 4개를 동시에 실행한다.
설정은 `cluster/config.json` 하나에 모았다. Mac에는 Python 3.9+, `kubectl`,
로그인된 Codex CLI가 필요하다. 학습 의존성은 Mac에 설치하지 않는다.

## 준비와 실행

Docker가 있는 머신에서 이 checkout으로 Linux amd64 이미지를 빌드·push한다.
Playground/Brax revision과 Python requirements, DINOv3 소스 revision을 고정하고,
Menagerie asset을 이미지에 포함한다. PVC의 ViT-L/16 가중치는 이미지에 복사하지 않는다.

```bash
IMAGE=YOUR_REGISTRY/map-all-you-need:rl-v1
docker buildx build --platform linux/amd64 -f cluster/Dockerfile \
  --build-arg CODE_REVISION="$(git rev-parse HEAD)" -t "$IMAGE" --push .
```

Mac에서 같은 checkout을 사용한다.

```bash
codex login status
python3 -m cluster login                # 수명 1시간 CPU pod, PVC 권한·가중치 검사
python3 -m cluster render --image "$IMAGE" # 제출 없이 30개 JSON manifest 생성
bash cluster/run.sh --image "$IMAGE"    # 사전 검사 → 백그라운드 감시/제출
```

`run.sh`가 출력한 `RUN_ID`를 이후 명령에 사용한다.
`render`는 이미지 없이도 구조를 확인할 수 있지만 `IMAGE_REQUIRED` placeholder를 제출하면 안 된다.
`run`과 `render`의 `--config path.json`은 기본 설정의 일부만 덮어쓴다.
`--run-id`는 선택 사항이다. 동일 ID는 `run`으로 다시 만들지 않고 `watch`로 재개한다.

```bash
python3 -m cluster status RUN_ID
python3 -m cluster watch RUN_ID         # 종료된 감시를 전경에서 재개
python3 -m cluster fetch RUN_ID         # 완료 결과 다운로드만 재시도
tail -f runs/cluster/RUN_ID/monitor.log
```

처음 실행하면 **같은 이미지의 GPU validation Job 1개**가 10개 환경 계약,
30개 조합의 짧은 학습·평가, 중단 후 checkpoint 복원, 옮긴 map 평가를 먼저 검사한다.
통과한 이미지의 실제 digest를 기록하고 이후 30개 Job을 그 digest로 고정한다.
검증 실패 시 본 학습 제출을 보류한다. 검증 Job은 30개 본 학습 Job과 별도다.
Warp/JAX/PyTorch의 GPU 메모리가 긴 테스트 세션에 누적되지 않도록
각 검증 case를 별도 Python 프로세스로 실행하고 종료 시 메모리를 회수한다.

현재 cluster 기본값: `nautilus / erl-ucsd`, PVC `sh-mapping`,
RTX A6000 1개(`nvidia.com/rtxa6000`), CPU 8, RAM 32Gi. 요청과 limit은 같다.
Private image는 `image_pull_secret`을 맞추고, public image만 쓰면 `null`로 설정할 수 있다.
DINO 소스는 이미지의 `/opt/dinov3`, 가중치는
`/mnt/dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`이다.

## 감시와 복구

Mac에서 `caffeinate -i`와 함께 controller가 실행된다. 기본 60초 간격으로
Job/Pod·학습 step을 확인하며 다운로드와 agent 판단 중에도 감시를 계속한다.
Mac의 전원 종료·인터넷 단절·덮개를 닫아 발생하는 잠자기는 막지 못한다.
Mac이 돌아오면 `watch`로 이어갈 수 있고, cluster 학습은 Mac과 독립적으로 계속된다.
잠금 파일과 Job UID를 사용하므로 같은 run의 controller는 하나만 작동한다.
`status`는 마지막으로 수집한 로컬 상태와 갱신 시각을 보여준다.

시작·실패·장기 정체·결과 회수 시 `codex exec --output-schema`를 호출한다.
`agent/`에 입력 증거, JSON 판단, 실행 로그가 남는다. Agent는 읽기 전용으로
진단하고 controller만 제한된 조치를 적용한다. 별도 API key는 필요하지 않으며
현재 Codex CLI 인증을 사용한다. 호출에는 계정의 사용량이 소모된다.
Mac 앱에 포함된 Codex CLI가 있으면 우선 사용하고, 없으면 PATH의 `codex`를 사용한다.
Homebrew CLI가 앱의 현재 모델을 지원하지 않는 경우를 피하기 위한 선택이며,
`codex_binary` 설정으로 원하는 실행 파일을 지정할 수 있다.

- 일시적 API/전송 오류는 같은 작업을 다시 시도한다.
- **Failed가 확정된 Job만** 새 attempt로 복구한다. 최대 3회 재시도한다.
  Kubernetes에도 `podReplacementPolicy: Failed`를 지정하여 기존 Pod가 완전히
  종료되기 전에 대체 Pod를 만드는 것을 막는다.
- Agent 진단에 따라 실패 노드를 제외하고, 실제 `OOMKilled`인 경우에만 RAM을
  기본값의 최대 2배로 늘린다. GPU 메모리·batch·코드·reward·학습량은 자동 변경하지 않는다.
- Pending/정체/Job 누락은 진단만 수행한다. 살아 있거나 소유권이 불확실한 작업을 복제하지 않는다.
- Agent 호출 실패 시 상태 수집은 계속하고 복구 판단을 기다린다. 수정이 필요한 오류는 `held`로 보고한다.
- 학습 완료 checkpoint가 있으면 평가 실패는 평가부터 재시도한다.
  다운로드 실패는 학습·평가를 다시 실행하지 않는다.

`watch`를 Ctrl-C로 종료해도 Job을 삭제하지 않는다. 진행 중인 다운로드/agent 호출은
정리될 때까지 기다린다. Agent가 보류한 실행의 증거는 `run.json`, `diagnostics/`,
`agent/`에서 확인한다. 코드 수정이 필요하면 수정한 이미지를 검증하는 새 run을 만든다.

## 결과와 학습 설정

```text
Cluster: /mnt/map-all-you-need/<run-id>/<task>/<policy>/seed0/attempt-<n>/
Mac:     runs/cluster/<run-id>/<task>/<policy>/seed0/attempt-<n>/
```

각 완료 attempt의 checkpoint 전체, 설정, 지표, 로그와 참조 map-cache를 회수한다.
`complete.json`의 SHA-256 목록으로 임시 디렉터리를 검증한 뒤 원자적으로 이동한다.
실패 attempt와 그 checkpoint는 PVC에 유지되고, 복구 이력/실패 진단은 Mac에도 남는다.
Map cache는 `/mnt/map-all-you-need/<run-id>/cache/<task>/seed<seed>/`에 분리하여
여러 seed나 run을 동시에 제출해도 생성 파일이 충돌하지 않는다.
회수한 디렉터리의 `map-cache/`를 먼저 사용하므로 cluster 절대 경로 없이 평가할 수 있다.
평가 자체에는 Linux/GPU 학습 의존성이 필요하다.

State/map 기본 설정은 유지한다. RGB는 native state 물리 위에 64×64 RGB만 제공하며
actor/critic 각각 Brax CNN을 사용한다. 기본 env 128, eval env 8, batch 16,
관측 정규화 비활성화이며 카메라 설정도 checkpoint metadata에 저장한다.
설정 파일 `train_args`에 benchmark CLI 옵션을 문자열 목록으로 지정하면
환경 수·batch 등 명시적인 실험 설정을 변경할 수 있다.

약 100만 step마다 공식 checkpoint를 저장한다. 실제 간격은 task의 PPO batch와
공식 reset 주기에 맞춰 metadata에 기록한다. 중간 평가는 끄고 완료 후 100 episode를 평가한다.
공식 Brax는 reset 묶음의 중간에 policy를 저장하는 API를 제공하지 않는다.
따라서 state 기본값을 보존하면 일부 간격은 더 길다. 예를 들어
`AlohaSinglePegInsertion/state`는 최소 **13,107,200 step**, PushCube/state는
**6,553,600 step**이다. RGB/map 기본 간격은 약 82만–102만 step이다.
실제 학습 step도 같은 단위로 올림되며 목표값과 실제값을 각각 보관한다.
복구는 정상적으로 읽히는 최신 checkpoint의 normalizer/actor/critic을 사용하여
`전체 목표 − 저장된 누적 step`만 학습한다. Optimizer와 RNG는 새로 초기화하며 이를 기록한다.
각 attempt의 checkpoint 숫자는 그 attempt의 step이고, metadata의 `resume_base_steps`를
더하면 누적 step이다. 기존 format-2 Cartesian RGB checkpoint의 평가도 유지한다.

## 검증

```bash
# Mac: 실제 제출 없이 운영 장애/재시작/회수 계약 검사
python3 -m unittest discover -s cluster -p 'test_*.py' -v
# 이미지 내부: GPU, EGL, 정확한 DINO 경로 필요
DINO_WEIGHTS=/mnt/dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
RUN_SPEC='{"kind":"validation","output":"/tmp/mayn-validation","config":{}}' \
  python -m cluster.worker
```

운영 설정은 [reachy-task Nautilus 설정](https://github.com/suk063/reachy-task/blob/main/cluster/config/nautilus.yaml),
agent 호출은 [OpenAI 공식 non-interactive 문서](https://learn.chatgpt.com/docs/non-interactive-mode#_top)를 참고했다.
Pod 교체 조건은 [Kubernetes 공식 Job 문서](https://kubernetes.io/docs/concepts/workloads/controllers/job/#delayed-creation-of-replacement-pods)를 따른다.
