# 붕어빵 인형 바구니 대회 전략 가이드 (SO-ARM101, 2/7 대회)

## 대회 개요
- **로봇**: lerobot so-arm101 (SO101 Follower)
- **태스크**: 흰색 3개 + 노란색 2개 붕어빵 인형 5개를 바구니에 넣기 (속도 경쟁)
- **바구니 위치**: 자유 선택
- **기한**: 2월 7일
- **하드웨어**: Apple M4 Pro Max, RAM 36GB (훈련 시 `policy.device=mps` 사용)
- **FPS**: **25 고정** (사용 카메라가 25fps만 지원. 녹화·훈련·시연 모두 `--dataset.fps=25`로 통일)

---

## 1. ACT 사용

### 1.1 ACT 선택 이유
1. **속도**: `chunk_size=100`, `n_action_steps=50`처럼 한 번 추론에 50~100 스텝 액션을 쓰면, 25Hz 기준 약 2~4초마다 한 번만 추론해도 됨.
2. **실기구 검증**: SO101/ALOHA 튜토리얼에서 ACT가 기본으로 사용됨.
3. **M4 환경**: ResNet18 백본 등으로 36GB RAM + MPS에서 훈련·추론이 현실적.

### 1.2 권장 훈련 명령 (ACT)

```bash
lerobot-train \
  --dataset.repo_id=<your_repo>/bun_basket \
  --policy.type=act \
  --output_dir=outputs/train/act_bun_basket \
  --job_name=act_bun_basket \
  --policy.device=mps \
  --policy.input_shapes="{\"observation.images.front\": [3, 96, 96]}" \
  --policy.n_action_steps=50 \
  --policy.chunk_size=100 \
  --wandb.enable=true
```

- `n_action_steps`를 너무 크면 반응이 느려지고, 너무 작으면 추론 횟수가 늘어남. 30~50 구간에서 실기로 조정 권장.

---

## 2. 시각적 강인함: Jittering (흑백 처리 미사용)

### 2.1 흑백 처리 미권장
- **상단 카메라**에서는 인형이 바닥과 잘 구분되지만, **팔 끝 카메라**에서는 조명 반사로 인형이 배경과 뭉개져 보일 수 있습니다.
- 흑백 변환은 이 경계를 더 모호하게 만들어, 특히 팔 끝 카메라에서 그립 판단을 어렵게 할 수 있으므로 **흑백 처리(그레이스케일)는 사용하지 않는 것을 권장**합니다.

### 2.2 Jittering으로 대회장 조명에 강인하게
- 훈련 시 **두 카메라(상단·팔 끝) 영상에 각각 무작위로 색감 변환**(밝기·대비·채도·색상 등)을 적용하면, 모델은 “카메라마다 색이 조금 달라도 **모양이 붕어빵이면 잡아야 한다**”는 식으로 학습합니다.
- 이렇게 하면 대회장의 **낯선 조명 환경**에서도 **시각적 강인함**을 얻을 수 있습니다.

### 2.3 설정 방법
- LeRobot의 `ImageTransformsConfig`는 기본적으로 **brightness, contrast, saturation, hue, sharpness, affine** 등을 지원합니다.
- 훈련 시 **image_transforms.enable=True**로 두고, 필요하면 각 변환의 강도(예: brightness (0.8, 1.2))만 조정하면 됩니다.

```bash
lerobot-train \
  --dataset.repo_id=local/bun_basket \
  --dataset.image_transforms.enable=True \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=act \
  ...
```

- `max_num_transforms`: 프레임마다 적용할 augmentation 개수 (기본 3). 2~4 구간에서 실험해 보면 됩니다.
- 녹화·시연은 **그대로 RGB 3채널**로 하면 되며, 별도 전처리 없이 훈련 시 Jittering만 켜 주면 됩니다.

---

## 3. control_rate(FPS) 조정 방법

### 3.1 LeRobot에서의 “제어 주기”
- **제어 주기 = 녹화 FPS = 추론 FPS**로 통일됩니다. 별도 `control_rate` 파라미터는 없고, **`dataset.fps`**가 그 역할을 합니다.

### 3.2 녹화 시 FPS 설정

- **현재 사용 중인 카메라는 25fps만 지원**하므로 `--dataset.fps=25`로 설정합니다.

```bash
lerobot-record \
  ... \
  --dataset.fps=25 \
  --dataset.repo_id=...
```

- `record_loop` 내부에서 `precise_sleep(max(1/fps - dt_s, 0.0))`로 **1/fps 초마다** 관측·액션 기록 및 로봇 제어.
- **현재 사용 중인 카메라는 25fps만 지원**하므로 **25**로 고정. 녹화·시연·훈련 데이터 모두 동일한 fps로 맞춥니다.

### 3.3 훈련·추론 시
- 데이터셋이 **25fps**로 기록되었으므로, 시연 시에도 **25fps**로 한 스텝씩 관측 → 정책 → 액션 전송을 하면 됩니다.
- ACT의 `n_action_steps`를 쓰면 “한 번 추론한 액션을 n_action_steps 스텝 동안 재생”하므로, 실제 추론 호출 주기는 `n_action_steps / fps` (초)가 됩니다.

정리:
- **녹화·시연 FPS**: `--dataset.fps=25` (사용 카메라 지원치로 고정).
- **실제 제어 주기**: 1/25초.
- **추론 주기**: ACT 기준 `n_action_steps / fps` (예: 50/25 = 2초마다 한 번).

---

## 4. Hugging Face vs 로컬 저장

### 4.1 비교

| 항목 | Hugging Face Hub | 로컬만 |
|------|------------------|--------|
| 녹화 직후 | `push_to_hub=True` 시 자동 업로드 (시간·네트워크 사용) | `push_to_hub=False`로 업로드 생략 |
| 훈련 | `repo_id`만 주면 캐시에서 또는 Hub에서 다운로드 | `repo_id` + `root`로 로컬 경로 지정 |
| 백업·공유 | Hub에 버전·공유 용이 | 없음 (로컬 백업은 직접) |
| 대회 준비 기간 | 업/다운로드 대기 가능 | 디스크만 있으면 즉시 훈련·재녹화 |

### 4.2 대회까지 일정이 촉박할 때 권장: 로컬 우선

1. **녹화**
   ```bash
   lerobot-record \
     ... \
     --dataset.repo_id=local/bun_basket \
     --dataset.push_to_hub=False
   ```
   - 데이터는 `~/.cache/huggingface/lerobot/local/bun_basket` (또는 `LEROBOT_HOME`이 있으면 그 아래)에만 저장.

2. **훈련**
   - 같은 로컬 경로를 쓰려면:
   ```bash
   lerobot-train \
     --dataset.repo_id=local/bun_basket \
     --dataset.root=~/.cache/huggingface/lerobot \
     ...
   ```
   - 또는 `repo_id`를 폴더 이름처럼 쓰고 `root`를 그 상위 경로로 두면, `root/repo_id`에서 읽습니다.

3. **나중에 공유/백업**
   - 훈련·대회가 끝난 뒤 `dataset.push_to_hub()` 또는 `huggingface-cli upload ...`로 올리면 됨.

정리: **2월 7일까지는 `--dataset.push_to_hub=False`로 로컬만 사용하고, 훈련도 `dataset.root`로 같은 로컬 경로를 지정**하는 구성을 추천합니다.

---

## 5. 에피소드 녹화 최적화

### 5.1 문서 권장치 (il_robots.mdx)
- **에피소드 수**: 최소 50개, 위치당 10개 수준으로 다양하게.
- **카메라**: 고정, 조작 물체가 항상 보이도록.
- **그래스핑**: 한 가지 방식으로 일관되게 녹화 후, 성능이 안정되면 그다음에 변형(위치·각도) 추가.
- **변형**: 한 번에 너무 많이 넣지 말고, 단계적으로.

### 5.2 이 대회에 맞춘 제안

1. **바구니 위치**
   - 한 번 정하면 **고정**. 모든 에피소드에서 동일 위치로 넣도록 녹화.

2. **붕어빵 배치**
   - 3~5곳 정도의 “시작 배치”를 정해 두고, **배치별로 10~15에피소드**씩 수집.
   - 예: 배치 A 12개, 배치 B 12개, 배치 C 12개 → 총 36개 이상.

3. **에피소드 길이**
   - 5개를 한 에피소드에 다 넣는다면: `--dataset.episode_time_s=90` 또는 `120`.
   - 한 에피소드당 1~2개만 넣고 “짧은 스킬”로 많이 모은 뒤 나중에 합치는 전략도 가능 (구현이 더 복잡할 수 있음).

4. **리셋 시간**
   - 리셋이 빠르면 `--dataset.reset_time_s=30` 정도로 줄여서 전체 녹화 시간 단축.

5. **녹화 명령 예시**

   ```bash
   lerobot-record \
     --robot.type=so101_follower \
     --robot.port=/dev/tty.usbmodemXXXX \
     --robot.id=my_follower \
     --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 25}}" \
     --teleop.type=so101_leader \
     --teleop.port=/dev/tty.usbmodemYYYY \
     --teleop.id=my_leader \
     --dataset.repo_id=local/bun_basket \
     --dataset.push_to_hub=False \
     --dataset.fps=25 \
     --dataset.num_episodes=60 \
     --dataset.episode_time_s=90 \
     --dataset.reset_time_s=30 \
     --dataset.single_task="Put all fish buns into the basket" \
     --display_data=true
   ```

6. **품질 관리**
   - 실패·흔들림이 많은 에피소드는 재녹화(← 키로 에피소드 취소 후 다시 녹화).
   - “카메라만 보고도 할 수 있는지”를 기준으로 하면, 나중에 정책이 카메라 입력만으로도 잘 맞추기 쉬움.

---

## 6. 일정 체크리스트 (2/7까지)

1. **설정 고정**: 바구니 위치, 카메라 위치, FPS(25, 카메라 지원치), 한 가지 그립/동선.
2. **데이터**: 최소 50~60 에피소드, 3~5가지 시작 배치로 분산.
3. **저장**: `push_to_hub=False`, 로컬 `repo_id`(예: `local/bun_basket`)로 통일.
4. **시각 강인함**: 훈련 시 `image_transforms.enable=True`로 Jittering 적용 (흑백 미사용).
5. **모델**: ACT 우선, `n_action_steps` 30~50으로 실기 테스트.
6. **훈련**: MPS 사용, `output_dir`/체크포인트 관리.
7. **시연**: 같은 FPS로 평가 에피소드 여러 번 돌려서 안정성 확인.

이 가이드를 기준으로 녹화 → 훈련 → 시연 파이프라인을 맞춘 뒤, Jittering·ACT chunk 크기·에피소드 수만 실기로 미세 조정하면 대회에 최적화하기 좋습니다.
