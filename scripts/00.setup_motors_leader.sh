#!/usr/bin/env bash
# 리더암(12v) 모터 설정 — SO101 Leader
# 포트: /dev/tty.usbmodem5AE60830721
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/01.setup_motors_leader.sh
#   또는  bash scripts/01.setup_motors_leader.sh
# 실행 순서: 모터 설정은 최초 1회 또는 포트/모터 변경 시만. 캘리브레이션(02, 03) 전에 실행.

set -e
cd "$(dirname "$0")/.."
lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5AE60830721
