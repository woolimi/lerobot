#!/usr/bin/env bash
# 팔로워암(5v) 캘리브레이션 — SO101 Follower
# 포트: /dev/tty.usbmodem5AE60810051  |  ID: my_follower_arm1
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/02.calibrate_follower.sh
#   또는  bash scripts/02.calibrate_follower.sh
# 실행 순서: 00, 01(모터 설정) 후 실행. 녹화·시연 시 같은 ID(my_follower_arm1) 사용해야 함.

set -e
cd "$(dirname "$0")/.."
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60810051 \
  --robot.id=my_follower_arm1
