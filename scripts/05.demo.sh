#!/usr/bin/env bash
# LeRobot 시연 — outputs/train/ 훈련 모델로 로봇 제어 (대회용)
# 모델을 먼저 로드한 뒤 엔터를 누르면 바로 동작. 텔레오프 없이 정책만 사용.
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/05.demo.sh
#   POLICY_PATH=outputs/train/v0.1 ./scripts/05.demo.sh
#
# 환경변수:
#   POLICY_PATH        모델 경로. 비우면 outputs/train/ 목록에서 선택
#   ROBOT_PORT, ROBOT_ID, CAMERA_*, DISPLAY_DATA
#   VISION_CONFIG_PATH 밝기/대비 등 vision 설정 YAML (기본: 02.record.sh 와 동일)
#   EPISODE_TIME_S     한 에피소드 길이(초). 비우면 60초. 예: EPISODE_TIME_S=20
#
# demo 모드: 녹화 없음, 엔터 후 시작, Ctrl+C 할 때까지 추론 반복.
# 녹화는 02.record.sh 사용.
#
# [추론 뷰 = 학습 뷰]
#   추론에 쓰이는 관측은 VISION_CONFIG_PATH(밝기/대비/감마, IBR 등)가 적용된 영상입니다.
#   학습 데이터를 02.record.sh로 같은 VISION_CONFIG_PATH로 녹화했다면 동일한 뷰로 학습·추론됩니다.
#
# [로봇 움직임 방해 없음]
#   매 프레임에서 액션을 먼저 보낸 뒤 화면 표시가 이루어져, 시각화가 제어를 막지 않습니다.
#   단, vision 설정에서 IBR(세그멘테이션)을 켜면 프레임당 연산이 늘어 루프가 느려질 수 있습니다.
#
# 종료: Ctrl+C

set -e
cd "$(dirname "$0")/.."

# Ultralytics(YOLO) 배너/반복 로그 억제
export YOLO_VERBOSE="${YOLO_VERBOSE:-false}"

ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5AE60810051}"
ROBOT_ID="${ROBOT_ID:-my_follower_arm1}"
CAMERA_TOP_INDEX="${CAMERA_TOP_INDEX:-1}"
CAMERA_GRIPPER_INDEX="${CAMERA_GRIPPER_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
CAMERA_FPS="${CAMERA_FPS:-30}"

DISPLAY_DATA="${DISPLAY_DATA:-false}"
VISION_CONFIG_PATH="${VISION_CONFIG_PATH:-configs/vision_dual_camera_example.yaml}"
# 한 에피소드 길이(초). 데모는 이 시간마다 루프 한 번 종료 후 즉시 다음 에피소드 시작.
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"

# POLICY_PATH 미설정 시 outputs/train/ 목록에서 선택
TRAIN_DIR="outputs/train"
if [[ -z "${POLICY_PATH:-}" ]]; then
  if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "Error: $TRAIN_DIR not found. Train a model first or set POLICY_PATH."
    exit 1
  fi
  RUNS=()
  for d in "$TRAIN_DIR"/*/; do
    [[ -d "$d" ]] && RUNS+=("${d%/}")
  done
  if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "Error: No model runs found in $TRAIN_DIR"
    exit 1
  fi
  echo "=== $TRAIN_DIR 에 있는 모델 ==="
  for i in "${!RUNS[@]}"; do
    echo "  $((i + 1))) $(basename "${RUNS[$i]}")"
  done
  echo ""
  echo -n "모델 선택 (번호 또는 경로 입력): "
  read -r CHOICE
  if [[ -z "$CHOICE" ]]; then
    echo "Error: 선택이 비어 있습니다."
    exit 1
  fi
  if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [[ "$CHOICE" -ge 1 ]] && [[ "$CHOICE" -le ${#RUNS[@]} ]]; then
    POLICY_PATH="${RUNS[$((CHOICE - 1))]}"
  else
    POLICY_PATH="$CHOICE"
  fi
  echo "선택: $POLICY_PATH"
  echo ""
fi

# POLICY_PATH 가 pretrained_model 이 아니면 체크포인트 선택/자동 선택
if [[ -d "$POLICY_PATH" ]]; then
  if [[ -d "${POLICY_PATH}/checkpoints" ]]; then
    CHECKPOINTS=()
    for d in "${POLICY_PATH}/checkpoints"/*/; do
      [[ -d "${d}pretrained_model" ]] && CHECKPOINTS+=("$(basename "$d")")
    done
    # 정렬 (002000, 004000, ... last 순)
    CHECKPOINTS=($(printf '%s\n' "${CHECKPOINTS[@]}" | sort))
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
      echo "Error: No pretrained_model found under ${POLICY_PATH}/checkpoints/"
      exit 1
    fi
    if [[ ${#CHECKPOINTS[@]} -eq 1 ]]; then
      POLICY_PATH="${POLICY_PATH}/checkpoints/${CHECKPOINTS[0]}/pretrained_model"
      echo "Checkpoint: ${CHECKPOINTS[0]} (유일)"
    else
      echo "=== 체크포인트 선택 ($(basename "$POLICY_PATH")) ==="
      for i in "${!CHECKPOINTS[@]}"; do
        echo "  $((i + 1))) ${CHECKPOINTS[$i]}"
      done
      echo ""
      echo -n "체크포인트 선택 (번호, 기본=마지막): "
      read -r CP_CHOICE
      if [[ -z "$CP_CHOICE" ]]; then
        CP_CHOICE=${#CHECKPOINTS[@]}
      fi
      if [[ "$CP_CHOICE" =~ ^[0-9]+$ ]] && [[ "$CP_CHOICE" -ge 1 ]] && [[ "$CP_CHOICE" -le ${#CHECKPOINTS[@]} ]]; then
        POLICY_PATH="${POLICY_PATH}/checkpoints/${CHECKPOINTS[$((CP_CHOICE - 1))]}/pretrained_model"
        echo "선택: ${CHECKPOINTS[$((CP_CHOICE - 1))]}"
      else
        echo "Error: 잘못된 선택입니다."
        exit 1
      fi
    fi
    echo ""
  elif [[ ! -f "${POLICY_PATH}/config.json" ]]; then
    echo "Error: POLICY_PATH is not a checkpoint dir (no config.json): $POLICY_PATH"
    exit 1
  fi
else
  echo "Error: POLICY_PATH not found: $POLICY_PATH"
  exit 1
fi

CAMERAS_JSON="{ top: {type: opencv, index_or_path: ${CAMERA_TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${CAMERA_GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"
DATASET_FPS="${CAMERA_FPS}"

echo "=== LeRobot Demo (Policy only, no recording — Ctrl+C to stop) ==="
echo "Policy:   $POLICY_PATH"
echo "Robot:    $ROBOT_PORT  id=$ROBOT_ID"
echo "Cameras:  top=${CAMERA_TOP_INDEX}, gripper=${CAMERA_GRIPPER_INDEX} (${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps)"
echo "Episode:  ${EPISODE_TIME_S}s per run (set EPISODE_TIME_S to change)"
echo "Vision:   ${VISION_CONFIG_PATH}"
echo ""

lerobot-record \
  --demo=true \
  --play_sounds=false \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS_JSON}" \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id=woolim/eval_demo_tmp \
  --dataset.single_task="Policy demo" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --display_data="${DISPLAY_DATA}" \
  --vision.config_path="${VISION_CONFIG_PATH}" \
  "$@"
