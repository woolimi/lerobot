#!/usr/bin/env bash

set -e
cd "$(dirname "$0")/.."

########################################
# ⭐ CUDA Memory Fragmentation 완화
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

########################################
# ⭐ Mixed Precision (LeRobot / Accelerate 경로)
########################################
export ACCELERATE_MIXED_PRECISION=bf16

DATA_DIR="${DATA_DIR:-./data}"

if [[ -z "${REPO_ID:-}" ]] || [[ -z "${DATASET_ROOT:-}" ]] || [[ -z "${MODEL_VERSION:-}" ]]; then
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: $DATA_DIR not found."
    exit 1
  fi

  CANDIDATES=()
  for d in "$DATA_DIR"/*/; do
    [[ -d "$d" ]] && [[ -f "${d}meta/info.json" ]] && CANDIDATES+=("$(basename "$d")")
  done

  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "No LeRobot datasets in $DATA_DIR."
    exit 1
  fi

  echo "=== 훈련 데이터셋 선택 ==="
  for i in "${!CANDIDATES[@]}"; do
    echo "  $((i + 1))) ${CANDIDATES[$i]}"
  done

  echo -n "데이터셋 번호: "
  read -r NUM

  SELECTED_FOLDER="${CANDIDATES[$((NUM - 1))]}"
  DATASET_ROOT="./data/${SELECTED_FOLDER}"
  REPO_ID="woolim/${SELECTED_FOLDER}"

  echo -n "모델 이름: "
  read -r MODEL_VERSION
  MODEL_VERSION="${MODEL_VERSION:-act_pilot_default}"
fi

MODEL_VERSION="${MODEL_VERSION:-act}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${MODEL_VERSION}}"
JOB_NAME="${JOB_NAME:-act_pilot_${MODEL_VERSION}}"

if [[ "$(uname -s)" == "Linux" ]]; then
  POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
else
  POLICY_DEVICE="${POLICY_DEVICE:-mps}"
fi

NUM_WORKERS="${NUM_WORKERS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo ""
echo "Dataset: repo_id=${REPO_ID} root=${DATASET_ROOT}"
echo "Model:   ${MODEL_VERSION}"
echo "Output:  ${OUTPUT_DIR}"
echo "Device:  ${POLICY_DEVICE}  num_workers=${NUM_WORKERS}"
echo ""

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device="${POLICY_DEVICE}" \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --dataset.image_transforms.enable=true \
  --wandb.enable=true \
  --steps=400000 \
  --batch_size=16 \
  --num_workers="${NUM_WORKERS}" \
  --save_checkpoint=true \
  --save_freq=3000 \
  --dataset.video_backend=torchcodec \
  --dataset.use_ibr_images=false \
  "$@"
