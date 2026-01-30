# SO-ARM101 설정 스크립트

리더암·팔로워암 포트가 이미 정해져 있으므로, 아래 스크립트로 setup/calibrate를 실행하면 됩니다.

## 포트

| 장치 | 포트 |
|------|------|
| 리더암 (12v) | `/dev/tty.usbmodem5AE60830721` |
| 팔로워암 (5v) | `/dev/tty.usbmodem5AE60810051` |

## 실행 순서

1. **모터 설정** (최초 1회 또는 모터/포트 변경 시)
   ```bash
   ./scripts/setup_motors_leader.sh    # 리더 먼저
   ./scripts/setup_motors_follower.sh # 팔로워
   ```

2. **캘리브레이션** (설정 후 또는 리더/팔로워 교체 시)
   ```bash
   ./scripts/calibrate_leader.sh
   ./scripts/calibrate_follower.sh
   ```

## 사용법

프로젝트 루트에서:

```bash
chmod +x scripts/*.sh
./scripts/setup_motors_leader.sh
./scripts/setup_motors_follower.sh
./scripts/calibrate_leader.sh
./scripts/calibrate_follower.sh
```

또는 `bash scripts/setup_motors_leader.sh` 처럼 실행해도 됩니다.

## ID

- 리더: `my_leader_arm`
- 팔로워: `my_follower_arm1`

녹화·시연 시 같은 ID를 사용해야 캘리브레이션이 적용됩니다.

---

## ACT 훈련 (data/first-try)

`data/first-try`에 녹화한 데이터로 ACT 모델을 학습할 때:

```bash
./scripts/train_act_first_try.sh
```

- **데이터**: `--dataset.repo_id=woolim/record_test` · `--dataset.root=./data/first-try`
- **정책**: ACT, `--policy.device=mps` (Mac)
- **출력**: `outputs/train/act_first_try/`
- **Jittering**: `image_transforms.enable=true` (기본)
- **WandB**: 기본 비활성. 사용하려면 `--wandb.enable=true` 추가

환경변수로 경로·이름을 바꿀 수 있습니다:

```bash
DATASET_ROOT=./data/other_dir OUTPUT_DIR=outputs/train/act_other ./scripts/train_act_first_try.sh
```

추가 인자는 그대로 `lerobot-train`에 전달됩니다:

```bash
./scripts/train_act_first_try.sh --steps=20000 --batch_size=16
```
