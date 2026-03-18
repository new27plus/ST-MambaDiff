#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="$(basename "$SCRIPT_DIR")"
DEFAULT_DATA_ROOT="/home/hp/code/Spatio-Temporal_U-Net_Denoising/Spatio-Temporal_U-Net_Denoising/datasets/NW-UCLA/"
DEFAULT_SEED=42

DATA_ROOT="$DEFAULT_DATA_ROOT"
SEED="$DEFAULT_SEED"
GPU=0
LOG_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$SCRIPT_DIR/logs/${EXPERIMENT_NAME}_${SEED}"
fi

CMD=(python "$SCRIPT_DIR/train.py" --data_root "$DATA_ROOT" --seed "$SEED" --gpu "$GPU")
if [[ -n "$LOG_DIR" ]]; then
  CMD+=(--log_dir "$LOG_DIR")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
