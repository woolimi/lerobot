#!/usr/bin/env bash
# 팔로워암(5v) 캘리브레이션 — SO101 Follower
# 포트: /dev/tty.usbmodem5AE60810051

set -e
cd "$(dirname "$0")/.."
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60810051 \
  --robot.id=my_follower_arm1
