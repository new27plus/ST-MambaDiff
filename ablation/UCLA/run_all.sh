#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_ROOT="/home/hp/code/Spatio-Temporal_U-Net_Denoising/Spatio-Temporal_U-Net_Denoising/datasets/NW-UCLA/"
DEFAULT_SEED=42

DATA_ROOT="$DEFAULT_DATA_ROOT"
SEED="$DEFAULT_SEED"
GPU=0
SUMMARY_FILE="ablation_final_summary.txt"
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
    --summary_file)
      SUMMARY_FILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

EXPERIMENTS=(
  "conv_only"
  "transformer_only"
  "mamba_only"
  "standard_ddpm"
  "neutral_decoder"
)

for exp in "${EXPERIMENTS[@]}"; do
  echo "Running experiment: ${exp}"
  bash "$SCRIPT_DIR/${exp}/run.sh" \
    --data_root "$DATA_ROOT" \
    --seed "$SEED" \
    --gpu "$GPU" \
    "${EXTRA_ARGS[@]}"
done

python "$SCRIPT_DIR/aggregate_results.py" \
  --ablation_root "$SCRIPT_DIR" \
  --output_dir "$SCRIPT_DIR" \
  --summary_file "$SUMMARY_FILE"

echo "All ablation experiments finished."
echo "Final summary: $SCRIPT_DIR/$SUMMARY_FILE"
