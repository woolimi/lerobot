#!/usr/bin/env bash
# 리더암(12v) 캘리브레이션 — SO101 Leader
# 포트: /dev/tty.usbmodem5AE60830721  |  ID: my_leader_arm
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/03.calibrate_leader.sh
#   또는  bash scripts/03.calibrate_leader.sh
# 실행 순서: 00, 01(모터 설정) 후 실행. 녹화·시연 시 같은 ID(my_leader_arm) 사용해야 함.

set -e
cd "$(dirname "$0")/.."
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5AE60830721 \
  --teleop.id=my_leader_arm
