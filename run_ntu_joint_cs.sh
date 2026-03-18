#!/bin/bash

# ==============================================================
# ⚙️ 全局配置区 (Global Configuration)
# ==============================================================
PYTHON="python"
# DATA_PATH="datasets/NTU60/ntu/NTU60_CS.npz" 
# SPLIT="CS"
# DATA_PATH="datasets/NTU60/ntu/NTU60_CV.npz" 
# SPLIT="CV"

# DATA_PATH="datasets/NTU120/ntu120/NTU120_CSub.npz" 
# SPLIT="CSub"
DATA_PATH="datasets/NTU120/ntu120/NTU120_CSet.npz" 
SPLIT="CSet"
BATCH_SIZE=64
WINDOW_SIZE=64 

MODALITIES=("joint" "bone" "motion")

# ==============================================================
# 🔄 循环开始 (Loop through modalities)
# ==============================================================

for MODALITY in "${MODALITIES[@]}"; do

    echo ""
    echo "################################################################"
    echo "🌊 STARTING PIPELINE FOR MODALITY: ${MODALITY} 🌊"
    echo "################################################################"
    echo ""

    # 定义 Checkpoint 路径
    CKPT_DIR="./checkpoints_ntu/${SPLIT}/${MODALITY}"

    # --------------------------------------------------------------
    # 🧹 0. 清理旧权重 (Safety Clean)
    # --------------------------------------------------------------
    # 这一步至关重要，防止 Joint 的权重混入 Bone，或者 Mamba/Conv 结构冲突
    echo "⚠️  [${MODALITY}] Cleaning old checkpoints in: $CKPT_DIR"
    rm -rf "$CKPT_DIR"/*
    mkdir -p "$CKPT_DIR"

    # --------------------------------------------------------------
    # 🚀 1. 运行 Phase 1: 预训练 (Pre-training)
    # --------------------------------------------------------------
    echo "=================================================="
    echo "🚀 [Phase 1] Pre-training: ${MODALITY}"
    echo "   Strategy: Physics-Consistent Denoising (T=1000)"
    echo "=================================================="

    $PYTHON phase1_ntu_pretrain.py \
        --data_path "$DATA_PATH" \
        --split "$SPLIT" \
        --modality "$MODALITY" \
        --batch_size "$BATCH_SIZE" \
        --epochs 80 \
        --lr 0.0001 \
        --patience 20

    if [ $? -ne 0 ]; then
        echo "❌ [${MODALITY}] Phase 1 Failed. Stopping pipeline for this modality."
        exit 1
    fi

    # --------------------------------------------------------------
    # 🚀 2. 运行 Phase 2: 分类微调 (Fine-tuning)
    # --------------------------------------------------------------
    echo ""
    echo "=================================================="
    echo "🚀 [Phase 2] SOTA Fine-tuning: ${MODALITY}"
    echo "   Strategy: SGD + Mixup + Label Smoothing"
    echo "   Note: Using LR=0.1 because Optimizer is SGD!"
    echo "=================================================="

    # 【关键调整】
    # 因为代码改用了 SGD (配合 Mixup)，LR 必须从 0.001 调大到 0.1
    # Weight Decay 使用 SGD 标准值 0.0004
    $PYTHON phase2_ntu_finetune.py \
        --data_path "$DATA_PATH" \
        --split "$SPLIT" \
        --modality "$MODALITY" \
        --batch_size "$BATCH_SIZE" \
        --epochs 120 \
        --lr 0.1 \
        --weight_decay 0.0004 \
        --warmup_epochs 5 \
        --alpha 0.2

    if [ $? -ne 0 ]; then
        echo "❌ [${MODALITY}] Phase 2 Failed. Stopping."
        exit 1
    fi

    echo ""
    echo "✅ [${MODALITY}] Finished Successfully!"
    echo "----------------------------------------------------------------"

done

echo ""
echo "🎉🎉🎉 ALL STREAMS COMPLETED! 🎉🎉🎉"
echo "You are now ready for Score Fusion (Ensemble)!"