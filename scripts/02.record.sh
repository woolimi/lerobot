#!/usr/bin/env bash
# LeRobot 데이터 수집 — 리더암으로 조작하면서 에피소드 녹화
# 텔레오프(01.teleop.sh)와 동일한 로봇/카메라 설정. dataset.repo_id, single_task 필요.
#
# 사용 예 (프로젝트 루트에서):
#   ./scripts/02.record.sh
#   SINGLE_TASK="큐브 잡아서 상자에 넣기" ./scripts/02.record.sh
#   NUM_EPISODES=5 REPO_ID=woolim/my_task ./scripts/02.record.sh
#
# 환경변수:
#   ROBOT_PORT         팔로워(로봇) 시리얼 포트
#   TELEOP_PORT        리더(조종) 시리얼 포트
#   ROBOT_ID / TELEOP_ID
#   CAMERA_TOP_INDEX / CAMERA_GRIPPER_INDEX / CAMERA_WIDTH / CAMERA_HEIGHT / CAMERA_FPS  (01.teleop.sh 와 동일)
#   REPO_ID            데이터셋 repo_id (기본: woolim/record). 훈련 시 같은 이름 사용
#   DATASET_ROOT       데이터 저장 로컬 경로. 비우면 ./data/<SINGLE_TASK> 사용 (공백·/·: 는 _ 로 치환)
#   SINGLE_TASK        태스크 설명. 비우면 실행 시 입력 프롬프트. 기본 데이터 폴더명으로도 사용
#   NUM_EPISODES       녹화할 에피소드 수 (기본: 25)
#   DISPLAY_DATA       화면에 카메라 표시 (기본: true)
#   PUSH_TO_HUB        녹화 후 Hub 업로드 (기본: false). true 로 올리기
#   VISION_CONFIG_PATH vision 설정 YAML 경로 (기본: configs/vision_dual_camera_example.yaml). 01.teleop.sh 와 동일 권장.
#
# 종료: Ctrl+C

set -e
cd "$(dirname "$0")/.."

ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5AE60810051}"
TELEOP_PORT="${TELEOP_PORT:-/dev/tty.usbmodem5AE60830721}"
ROBOT_ID="${ROBOT_ID:-my_follower_arm1}"
TELEOP_ID="${TELEOP_ID:-my_leader_arm}"
CAMERA_TOP_INDEX="${CAMERA_TOP_INDEX:-0}"
CAMERA_GRIPPER_INDEX="${CAMERA_GRIPPER_INDEX:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
CAMERA_FPS="${CAMERA_FPS:-30}"

REPO_ID="${REPO_ID:-woolim/record}"
SINGLE_TASK="${SINGLE_TASK:-}"
NUM_EPISODES="${NUM_EPISODES:-40}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
VISION_CONFIG_PATH="${VISION_CONFIG_PATH:-configs/vision_dual_camera_example.yaml}"

if [[ -z "$SINGLE_TASK" ]]; then
  echo -n "SINGLE_TASK (태스크 설명): "
  read -r SINGLE_TASK
  if [[ -z "$SINGLE_TASK" ]]; then
    echo "Error: SINGLE_TASK is required."
    exit 1
  fi
fi

# DATASET_ROOT 미지정 시 SINGLE_TASK 로 폴더명 생성 (공백·/·: → _)
if [[ -z "${DATASET_ROOT:-}" ]]; then
  TASK_DIR="${SINGLE_TASK// /_}"
  TASK_DIR="${TASK_DIR//\//_}"
  TASK_DIR="${TASK_DIR//:/_}"
  DATASET_ROOT="./data/${TASK_DIR}"
fi

# 카메라 설정 — 01.teleop.sh 와 동일 (1280x720 @ 30fps)
CAMERAS_JSON="{ top: {type: opencv, index_or_path: ${CAMERA_TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${CAMERA_GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

# dataset.fps 는 카메라 fps 와 맞춤
DATASET_FPS="${CAMERA_FPS}"

echo "=== LeRobot Record ==="
echo "Robot:   $ROBOT_PORT  id=$ROBOT_ID"
echo "Teleop:  $TELEOP_PORT  id=$TELEOP_ID"
echo "Cameras: top=${CAMERA_TOP_INDEX}, gripper=${CAMERA_GRIPPER_INDEX} (${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps)"
echo "Dataset: repo_id=${REPO_ID}  root=${DATASET_ROOT}  num_episodes=${NUM_EPISODES}"
echo "Task:    ${SINGLE_TASK}"
echo "Push to Hub: ${PUSH_TO_HUB}"
echo "Vision config: ${VISION_CONFIG_PATH}"
echo ""

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS_JSON}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data="${DISPLAY_DATA}" \
  --vision.config_path="${VISION_CONFIG_PATH}" \
  "$@"
