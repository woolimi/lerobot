#!/usr/bin/env bash
# data/first-try 데이터로 ACT 모델 훈련 (LeRobot)
# 카메라: top, gripper | FPS: 30 | 에피소드: 10

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT="${DATASET_ROOT:-./data/first-try}"
REPO_ID="${REPO_ID:-woolim/record_test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_first_try}"
JOB_NAME="${JOB_NAME:-act_first_try}"

echo "Dataset: repo_id=${REPO_ID} root=${DATASET_ROOT}"
echo "Output:  ${OUTPUT_DIR}"
echo ""

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device=mps \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --dataset.image_transforms.enable=true \
  --wandb.enable=false \
  "$@"
