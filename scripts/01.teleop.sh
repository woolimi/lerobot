#!/usr/bin/env bash
# LeRobot 텔레오퍼레이션 — 리더암으로 팔로워암 제어
# 리더(Leader)를 움직이면 팔로워(Follower)가 따라 움직입니다.
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/01.teleop.sh
#
# 환경변수:
#   ROBOT_PORT         팔로워(로봇) 시리얼 포트
#   TELEOP_PORT        리더(조종) 시리얼 포트
#   ROBOT_ID           팔로워 id (기본: my_follower_arm1)
#   TELEOP_ID          리더 id (기본: my_leader_arm)
#   CAMERA_TOP_INDEX   top 카메라 인덱스 (기본: 0)
#   CAMERA_GRIPPER_INDEX  gripper 카메라 인덱스 (기본: 1)
#   CAMERA_WIDTH       해상도 가로 (기본: 1280). 30fps 쓰려면 1280x720 유지
#   CAMERA_HEIGHT      해상도 세로 (기본: 720). 640x480 쓰면 fps 25만 되는 경우 있음
#   CAMERA_FPS         카메라 fps (기본: 30). 640x480 일 때만 25로 낮추는 경우 있음
#   DISPLAY_DATA       화면에 카메라 표시 (기본: true)
#   VISION_CONFIG_PATH vision 설정 YAML 경로 (기본: configs/vision_dual_camera_example.yaml). gripper IBR, top RAW.
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
DISPLAY_DATA="${DISPLAY_DATA:-true}"
VISION_CONFIG_PATH="${VISION_CONFIG_PATH:-configs/vision_dual_camera_example.yaml}"

# 기본 1280x720 @ 30fps (find-cameras 기본 프로필과 동일 → 30fps 동작)
CAMERAS_JSON="{ top: {type: opencv, index_or_path: ${CAMERA_TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${CAMERA_GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

echo "=== LeRobot Teleop ==="
echo "Robot (follower): $ROBOT_PORT  id=$ROBOT_ID"
echo "Teleop (leader):  $TELEOP_PORT  id=$TELEOP_ID"
echo "Cameras:           top=${CAMERA_TOP_INDEX}, gripper=${CAMERA_GRIPPER_INDEX} (${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps)"
echo "Display:           $DISPLAY_DATA"
echo "Vision config:     $VISION_CONFIG_PATH (gripper IBR, top RAW)"
echo ""

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS_JSON}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --display_data="${DISPLAY_DATA}" \
  --vision.config_path="${VISION_CONFIG_PATH}" \
  "$@"
