#!/usr/bin/env bash
# 체크포인트에서 훈련 이어하기 (04.train.sh 로 저장된 체크포인트 사용)
# outputs/train/ 에서 런 선택 → 체크포인트(스텝) 선택 → 이어서 훈련
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/04.train_resume.sh
#   (런 선택 → 체크포인트 번호 선택)
#
# 환경변수로 건너뛰기:
#   CONFIG_PATH=... NEW_DATASET=fishes_pos ./scripts/04.train_resume.sh
#
# 새 데이터로 이어 훈련: 체크포인트 선택 후 "새 데이터셋" 프롬프트에서 data/ 폴더 선택.
#   새 데이터 선택 시 스텝을 0으로 초기화(--resume_reset_steps=true) 해서 추가로 steps 만큼 훈련.
#   이때 "새 모델명"을 물어보며, 입력하면 outputs/train/<모델명> 에 새 런으로 저장.
#
# 종료: Ctrl+C

set -e
cd "$(dirname "$0")/.."

TRAIN_DIR="${TRAIN_DIR:-outputs/train}"

# CONFIG_PATH 가 있으면 그대로 사용 (config 설정 그대로 이어받음)
if [[ -n "${CONFIG_PATH:-}" ]]; then
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: CONFIG_PATH not found: $CONFIG_PATH"
    exit 1
  fi
  EXTRA_ARGS=()
  if [[ -n "${NEW_DATASET:-}" ]]; then
    DATA_DIR="${DATA_DIR:-./data}"
    DATASET_ROOT="${DATA_DIR}/${NEW_DATASET}"
    REPO_PREFIX="${REPO_PREFIX:-woolim}"
    if [[ -d "$DATASET_ROOT" ]] && [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
      EXTRA_ARGS=(--dataset.repo_id="${REPO_PREFIX}/${NEW_DATASET}" --dataset.root="${DATASET_ROOT}" --resume_reset_steps=true)
      echo -n "새 모델명 (출력 폴더명, 비우면 기존 output_dir 그대로): "
      read -r NEW_MODEL_NAME
      if [[ -n "$NEW_MODEL_NAME" ]]; then
        EXTRA_ARGS+=(--output_dir="outputs/train/${NEW_MODEL_NAME}" --job_name="act_pilot_${NEW_MODEL_NAME}")
      fi
    fi
  fi
  echo "Resume from: $CONFIG_PATH"
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "Override: ${EXTRA_ARGS[*]}"
  echo ""
  lerobot-train --resume=true --config_path="${CONFIG_PATH}" "${EXTRA_ARGS[@]}" "$@"
  exit 0
fi

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "Error: $TRAIN_DIR not found. 먼저 04.train.sh 로 훈련하세요."
  exit 1
fi

# outputs/train/ 아래 런 폴더 목록 (checkpoints 가 있는 것만)
RUNS=()
for d in "$TRAIN_DIR"/act_pilot_*/; do
  [[ -d "$d" ]] && [[ -d "${d}checkpoints" ]] && RUNS+=("$(basename "$d")")
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
  # act_pilot_ 뿐 아니라 아무 런 폴더
  for d in "$TRAIN_DIR"/*/; do
    [[ -d "$d" ]] && [[ -d "${d}checkpoints" ]] && RUNS+=("$(basename "$d")")
  done
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "No run with checkpoints in $TRAIN_DIR. 04.train.sh 로 훈련 후 체크포인트를 만드세요."
  exit 1
fi

echo "=== 이어서 훈련할 런 선택 ==="
for i in "${!RUNS[@]}"; do
  echo "  $((i + 1))) ${RUNS[$i]}"
done
echo ""
echo -n "런 번호: "
read -r RUN_NUM
if [[ ! "$RUN_NUM" =~ ^[0-9]+$ ]] || [[ "$RUN_NUM" -lt 1 ]] || [[ "$RUN_NUM" -gt ${#RUNS[@]} ]]; then
  echo "Error: 잘못된 번호."
  exit 1
fi
RUN_NAME="${RUNS[$((RUN_NUM - 1))]}"
RUN_DIR="${TRAIN_DIR}/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"

# 체크포인트(스텝) 목록
STEPS=()
for d in "$CKPT_DIR"/*/; do
  if [[ -d "$d" ]] && [[ -f "${d}pretrained_model/train_config.json" ]]; then
    STEPS+=("$(basename "$d")")
  fi
done

# 숫자 순 정렬 (005000, 010000 ...)
STEPS=($(printf '%s\n' "${STEPS[@]}" | sort -n))

if [[ ${#STEPS[@]} -eq 0 ]]; then
  echo "Error: No checkpoint with train_config.json in $CKPT_DIR"
  exit 1
fi

echo ""
echo "=== 이어갈 체크포인트 (스텝) 선택 ==="
for i in "${!STEPS[@]}"; do
  echo "  $((i + 1))) ${STEPS[$i]}"
done
echo ""
echo -n "체크포인트 번호: "
read -r STEP_NUM
if [[ ! "$STEP_NUM" =~ ^[0-9]+$ ]] || [[ "$STEP_NUM" -lt 1 ]] || [[ "$STEP_NUM" -gt ${#STEPS[@]} ]]; then
  echo "Error: 잘못된 번호."
  exit 1
fi
STEP_NAME="${STEPS[$((STEP_NUM - 1))]}"
CONFIG_PATH="${RUN_DIR}/checkpoints/${STEP_NAME}/pretrained_model/train_config.json"

# 새 데이터셋 선택 (비우면 config 그대로)
DATA_DIR="${DATA_DIR:-./data}"
EXTRA_ARGS=()
if [[ -n "${NEW_DATASET:-}" ]]; then
  DATASET_ROOT="${DATA_DIR}/${NEW_DATASET}"
  REPO_PREFIX="${REPO_PREFIX:-woolim}"
  if [[ -d "$DATASET_ROOT" ]] && [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
    EXTRA_ARGS=(--dataset.repo_id="${REPO_PREFIX}/${NEW_DATASET}" --dataset.root="${DATASET_ROOT}" --resume_reset_steps=true)
    if [[ -z "${NEW_MODEL_NAME:-}" ]]; then
      echo -n "새 모델명 (출력 폴더명, 비우면 기존 output_dir 그대로): "
      read -r NEW_MODEL_NAME
    fi
    if [[ -n "${NEW_MODEL_NAME:-}" ]]; then
      EXTRA_ARGS+=(--output_dir="outputs/train/${NEW_MODEL_NAME}" --job_name="act_pilot_${NEW_MODEL_NAME}")
    fi
  else
    echo "Warning: NEW_DATASET=$NEW_DATASET not found or invalid, using config dataset."
  fi
elif [[ -d "$DATA_DIR" ]]; then
  CANDIDATES=()
  for d in "$DATA_DIR"/*/; do
    [[ -d "$d" ]] && [[ -f "${d}meta/info.json" ]] && CANDIDATES+=("$(basename "$d")")
  done
  if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
    echo ""
    echo "=== 새 데이터로 이어 훈련할까요? (비우면 config 데이터 그대로) ==="
    for i in "${!CANDIDATES[@]}"; do echo "  $((i + 1))) ${CANDIDATES[$i]}"; done
    echo "  0) config 데이터 그대로"
    echo ""
    echo -n "데이터셋 번호 또는 폴더명 (비우면 그대로): "
    read -r DS_CHOICE
    if [[ -n "$DS_CHOICE" ]]; then
      if [[ "$DS_CHOICE" == "0" ]]; then
        :
      elif [[ "$DS_CHOICE" =~ ^[0-9]+$ ]] && [[ "$DS_CHOICE" -ge 1 ]] && [[ "$DS_CHOICE" -le ${#CANDIDATES[@]} ]]; then
        NEW_DATASET="${CANDIDATES[$((DS_CHOICE - 1))]}"
        DATASET_ROOT="${DATA_DIR}/${NEW_DATASET}"
        REPO_PREFIX="${REPO_PREFIX:-woolim}"
        EXTRA_ARGS=(--dataset.repo_id="${REPO_PREFIX}/${NEW_DATASET}" --dataset.root="${DATASET_ROOT}" --resume_reset_steps=true)
        echo -n "새 모델명 (출력 폴더명, 비우면 기존 output_dir 그대로): "
        read -r NEW_MODEL_NAME
        if [[ -n "$NEW_MODEL_NAME" ]]; then
          EXTRA_ARGS+=(--output_dir="outputs/train/${NEW_MODEL_NAME}" --job_name="act_pilot_${NEW_MODEL_NAME}")
        fi
      else
        NEW_DATASET="$DS_CHOICE"
        DATASET_ROOT="${DATA_DIR}/${NEW_DATASET}"
        REPO_PREFIX="${REPO_PREFIX:-woolim}"
        if [[ -d "$DATASET_ROOT" ]] && [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
          EXTRA_ARGS=(--dataset.repo_id="${REPO_PREFIX}/${NEW_DATASET}" --dataset.root="${DATASET_ROOT}" --resume_reset_steps=true)
          echo -n "새 모델명 (출력 폴더명, 비우면 기존 output_dir 그대로): "
          read -r NEW_MODEL_NAME
          if [[ -n "$NEW_MODEL_NAME" ]]; then
            EXTRA_ARGS+=(--output_dir="outputs/train/${NEW_MODEL_NAME}" --job_name="act_pilot_${NEW_MODEL_NAME}")
          fi
        else
          echo "Warning: $NEW_DATASET not found under $DATA_DIR, using config dataset."
        fi
      fi
    fi
  fi
fi

echo ""
echo "Resume from: $CONFIG_PATH"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "Override: ${EXTRA_ARGS[*]}"
echo ""

lerobot-train --resume=true --config_path="${CONFIG_PATH}" "${EXTRA_ARGS[@]}" "$@"
