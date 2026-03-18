#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_ROOT="/home/hp/code/Spatio-Temporal_U-Net_Denoising/Spatio-Temporal_U-Net_Denoising/datasets/NW-UCLA/"
DEFAULT_SEED=42

DATA_ROOT="$DEFAULT_DATA_ROOT"
SEED="$DEFAULT_SEED"
GPU=0
LOG_DIR=""

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
      echo "Unsupported argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$SCRIPT_DIR/logs/sanity_check_${SEED}"
fi

python "$SCRIPT_DIR/neutral_decoder/train.py" \
  --data_root "$DATA_ROOT" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --log_dir "$LOG_DIR" \
  --subset_ratio 0.1 \
  --pretrain_epochs 2 \
  --finetune_epochs 2 \
  --num_workers 0

echo "Sanity check passed"
