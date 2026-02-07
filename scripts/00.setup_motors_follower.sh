#!/usr/bin/env bash
# 팔로워암(5v) 모터 설정 — SO101 Follower
# 포트: /dev/tty.usbmodem5AE60810051
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/00.setup_motors_follower.sh
#   또는  bash scripts/00.setup_motors_follower.sh
# 실행 순서: 리더(01)보다 먼저 해도 되고, 모터 설정은 최초 1회 또는 포트/모터 변경 시만.

set -e
cd "$(dirname "$0")/.."
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60810051
