#!/usr/bin/env bash
# LeRobot 카메라 점검 — 연결된 OpenCV/RealSense 카메라 목록 및 테스트 캡처
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/05.camera_check.sh              # 모든 카메라 검색 + 짧은 캡처
#   ./scripts/05.camera_check.sh opencv      # OpenCV 카메라만
#   ./scripts/05.camera_check.sh realsense   # RealSense 카메라만
#
# 출력: 콘솔에 카메라 목록, 이미지는 outputs/camera_check/ 에 저장

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/camera_check}"
RECORD_TIME_S="${RECORD_TIME_S:-2}"

echo "=== LeRobot Camera Check ==="
echo "Output dir: $OUTPUT_DIR"
echo "Capture duration: ${RECORD_TIME_S}s"
echo ""

if [[ -n "${1:-}" ]]; then
  lerobot-find-cameras "$1" --output-dir "$OUTPUT_DIR" --record-time-s "$RECORD_TIME_S"
else
  lerobot-find-cameras --output-dir "$OUTPUT_DIR" --record-time-s "$RECORD_TIME_S"
fi

echo ""
echo "Done. Images saved to $OUTPUT_DIR"
