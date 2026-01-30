#!/usr/bin/env bash
# 리더암(12v) 캘리브레이션 — SO101 Leader
# 포트: /dev/tty.usbmodem5AE60830721

set -e
cd "$(dirname "$0")/.."
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5AE60830721 \
  --teleop.id=my_leader_arm
