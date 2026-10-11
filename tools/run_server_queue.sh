#!/usr/bin/env bash
# =============================================================================
# ST-MambaDiff 返修 · 服务器一键实验队列（对应 EXPERIMENTS.md v3）
# =============================================================================
# 用法（服务器上，Experment 根目录）:
#   tmux new -s rev "bash tools/run_server_queue.sh 2>&1 | tee runs_server_queue.log"
#   # 含可选块（E8 种子、E4 种子 1/2）:
#   INCLUDE_OPTIONAL=1 bash tools/run_server_queue.sh 2>&1 | tee runs_server_queue.log
#
# 设计:
#   - 按依赖顺序执行: 冒烟 → E10/E11(便宜且表格必填) → E8c/E9(命运实验) → E4 → E5 → E6 → E13
#   - 每步完成后核验产物文件是否存在，失败立即停（避免空跑烧卡时）
#   - 断点续跑: 已有产物的步骤自动跳过（中断后重跑同一命令即可续）
#   - E8c/E9 合并: 预训练臂(ntu60_nd_reuse)即 E9 的 ND 侧，一次运行两用
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."          # Experment 根目录
ROOT=$(pwd)
QLOG="runs/server_queue_$(date +%Y%m%d-%H%M%S).log"
mkdir -p runs
OPTIONAL=${INCLUDE_OPTIONAL:-0}

log()  { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }
fail() { log "❌ $1 —— 队列中止"; exit 1; }
have() { [ -e "$1" ]; }

# ---- 前置检查 ---------------------------------------------------------------
log "=== 前置检查 ==="
python -c "import torch; assert torch.cuda.is_available()" || fail "CUDA 不可用"
python -c "import mamba_ssm, diffusers, sklearn, yaml, thop" 2>/dev/null || fail "依赖缺失: pip install diffusers scikit-learn pyyaml thop"
[ -d datasets/NW-UCLA/all_sqe ] || fail "缺 NW-UCLA 数据（见 EXPERIMENTS.md §1.2）"
[ -f datasets/NTU60/ntu/NTU60_CS.npz ] || fail "缺 NTU60_CS.npz"
log "前置检查通过（GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')）"

# ---- 0. E7 冒烟 --------------------------------------------------------------
log "=== [0] E7 NTU 冒烟（1 epoch） ==="
if ! ls runs/smoke_ntu 2>/dev/null; then
  python tools/pretrain_ntu.py --config configs/ntu60_cs.yaml --modality joint \
      --experiment smoke_ntu --set phase1.epochs=1 >> "$QLOG" 2>&1 \
      && rm -rf runs/smoke_ntu && log "冒烟通过，已清理" \
      || fail "E7 冒烟失败（查 $QLOG）"
else
  log "冒烟目录已存在，跳过（如需重跑请删 runs/smoke_ntu）"
fi

# ---- 1. E10 NTU 四协议无泄露融合（纯推理，先出必填表格） ------------------------
# 注: 若未上传 NTU120 npz（默认不上传），CSub/CSet 两协议自动跳过 → 回本地跑（见 EXPERIMENTS.md §3-E10）
log "=== [1] E10 NTU 四协议无泄露融合 ==="
for pair in "ntu60_cs ntu60_cs datasets/NTU60/ntu/NTU60_CS.npz" \
            "ntu60_cv ntu60_cv datasets/NTU60/ntu/NTU60_CV.npz" \
            "ntu120_csub ntu120_csub datasets/NTU120/ntu120/NTU120_CSub.npz" \
            "ntu120_cset ntu120_cset datasets/NTU120/ntu120/NTU120_CSet.npz"; do
  set -- $pair; CFG=$1; CKROOT=$2; NPZ=$3
  if have "runs/${CFG}/fusion_leakfree/fusion_report.txt"; then log "E10 ${CFG} 已有报告，跳过"; continue; fi
  if ! have "$NPZ"; then log "E10 ${CFG} 跳过（服务器无 ${NPZ}）→ 请回本地跑"; continue; fi
  have "artifacts/checkpoints/${CKROOT}/joint/phase2_sota_best.pt" || fail "缺 ${CKROOT} checkpoint"
  python tools/fusion_ntu_leakfree.py --config configs/${CFG}.yaml \
      --ckpt_root artifacts/checkpoints/${CKROOT} >> "$QLOG" 2>&1 \
      && log "E10 ${CFG} ✅" || fail "E10 ${CFG} 失败"
done

# ---- 2. E11 效率重测 ---------------------------------------------------------
log "=== [2] E11 效率统一重测 ==="
if have runs/efficiency/efficiency_report.txt; then log "已有报告，跳过"
else python tools/measure_efficiency.py >> "$QLOG" 2>&1 && log "E11 ✅" || fail "E11 失败"; fi

# ---- 3. E8c+E9 NTU60 两臂（命运实验） ------------------------------------------
log "=== [3] E8c+E9 NTU60-CS 两臂（预训练复用归档 phase1 / scratch） ==="
have artifacts/checkpoints/ntu60_cs/joint/phase1_best_acc.pt || fail "缺归档 phase1"
# 预训练臂（=E9 ND 侧）
if ls runs/ntu60_nd_reuse/joint/seed0/*/checkpoints/phase2_sota_best.pt >/dev/null 2>&1; then
  log "E8c 预训练臂已完成，跳过"
else
  python tools/finetune_ntu.py --config configs/ntu60_cs.yaml --modality joint \
      --experiment ntu60_nd_reuse --seed 0 \
      --phase1_ckpt artifacts/checkpoints/ntu60_cs/joint/phase1_best_acc.pt >> "$QLOG" 2>&1 \
      && log "E8c 预训练臂 ✅" || fail "E8c 预训练臂失败"
fi
# scratch 臂
if ls runs/ntu60_scratch/joint/seed0/*/checkpoints/phase2_sota_best.pt >/dev/null 2>&1; then
  log "E8c scratch 臂已完成，跳过"
else
  python tools/finetune_ntu.py --config configs/ntu60_cs.yaml --modality joint \
      --experiment ntu60_scratch --seed 0 >> "$QLOG" 2>&1 \
      && log "E8c scratch 臂 ✅" || fail "E8c scratch 臂失败"
fi
log "⚠️ [阶段1回传点] 建议此时打包回传 runs/ntu60_* 与 runs/ntu1{20,20}_*/ 产物"

# ---- 4. E9 标准 DDPM 侧（NTU） -------------------------------------------------
log "=== [4] E9 NTU 标准 DDPM 侧（全管线） ==="
if ls runs/ntu60_std_ddpm/joint/seed0/*/checkpoints/phase2_sota_best.pt >/dev/null 2>&1; then
  log "已完成，跳过"
else
  python tools/pretrain_ntu.py --config configs/ntu60_cs.yaml --modality joint \
      --experiment ntu60_std_ddpm --seed 0 --set model.decoder_time_condition=true >> "$QLOG" 2>&1 \
      && python tools/finetune_ntu.py --config configs/ntu60_cs.yaml --modality joint \
         --experiment ntu60_std_ddpm --seed 0 >> "$QLOG" 2>&1 \
      && log "E9 ✅" || fail "E9 失败"
fi

# ---- 5. E4 UCLA 标准 DDPM（seed0 phase2 本地完成，服务器跳过；只有跑可选种子才进这里） ----
if [ "$OPTIONAL" != "1" ]; then
  log "=== [5] E4 跳过（seed0 phase2 由本地完成；服务器仅当 INCLUDE_OPTIONAL=1 时补种子 1/2） ==="
fi
if [ "$OPTIONAL" = "1" ]; then
  log "=== [5] E4 可选：UCLA std_ddpm 种子 1/2（phase1+phase2 全管线） ==="
  for s in 1 2; do
    if ls runs/ucla_std_ddpm/joint/seed$s/*/checkpoints/phase2_joint_best_classifier.pt >/dev/null 2>&1; then
      log "E4 seed$s 已完成，跳过"; continue
    fi
    python tools/pretrain_ucla.py --config configs/ucla_final_arch.yaml --modality joint \
        --experiment ucla_std_ddpm --seed $s --set model.decoder_time_condition=true >> "$QLOG" 2>&1 \
        && python tools/finetune_ucla.py --config configs/ucla_final_arch.yaml \
           --modality joint --experiment ucla_std_ddpm --seed $s >> "$QLOG" 2>&1 \
        && log "E4 seed$s ✅" || fail "E4 seed$s 失败"
  done
fi

# ---- 6. E5 模态×算子 6 格 ------------------------------------------------------
log "=== [6] E5 模态×算子矩阵（dwconv/transformer × joint/bone/motion） ==="
for op in dwconv transformer; do
  for m in joint bone motion; do
    P1="runs/grid_${op}_${m}/${m}/seed0"
    if ls $P1/*/checkpoints/phase2_${m}_best_classifier.pt >/dev/null 2>&1; then
      log "E5 ${op}/${m} 已完成，跳过"; continue
    fi
    python tools/pretrain_ucla.py --config configs/ucla_final_arch.yaml \
        --modality $m --experiment grid_${op}_${m} --seed 0 \
        --set model.temporal_operator=${op} >> "$QLOG" 2>&1 \
        || fail "E5 ${op}/${m} pretrain 失败"
    python tools/finetune_ucla.py --config configs/ucla_final_arch.yaml \
        --modality $m --experiment grid_${op}_${m} --seed 0 >> "$QLOG" 2>&1 \
        || fail "E5 ${op}/${m} finetune 失败"
    log "E5 ${op}/${m} ✅"
  done
done

# ---- 7. E6 目标对比（可砍块） --------------------------------------------------
log "=== [7] E6 MAE / 单步目标对比 ==="
for obj in mae single_step; do
  if ls runs/ucla_${obj}/joint/seed0/*/checkpoints/phase2_joint_best_classifier.pt >/dev/null 2>&1; then
    log "E6 ${obj} 已完成，跳过"; continue
  fi
  python tools/pretrain_ucla.py --config configs/ucla_final_arch.yaml \
      --modality joint --experiment ucla_${obj} --seed 0 \
      --set phase1.objective=${obj} >> "$QLOG" 2>&1 \
      && python tools/finetune_ucla.py --config configs/ucla_final_arch.yaml \
         --modality joint --experiment ucla_${obj} --seed 0 >> "$QLOG" 2>&1 \
      && log "E6 ${obj} ✅" || log "E6 ${obj} 失败（继续队列，可后补）"
done

# ---- 8. E13 双曲线 -------------------------------------------------------------
log "=== [8] E13 去噪 MSE-t 双曲线 ==="
python tools/denoise_curve.py --preset ucla \
    --ckpt "$(ls runs/ucla_final/joint/seed0/*/checkpoints/phase1_joint_best_acc.pt | head -1)" \
    --label ND >> "$QLOG" 2>&1 && log "E13 ND ✅" || log "E13 ND 失败（可后补）"
python tools/denoise_curve.py --preset ucla \
    --ckpt "$(ls runs/ucla_std_ddpm/joint/seed0/*/checkpoints/phase1_joint_best_acc.pt | head -1)" \
    --label StdDDPM >> "$QLOG" 2>&1 && log "E13 StdDDPM ✅" || log "E13 StdDDPM 失败（可后补）"

# ---- 9. 可选块 -----------------------------------------------------------------
if [ "$OPTIONAL" = "1" ]; then
  log "=== [9] 可选块（INCLUDE_OPTIONAL=1） ==="
  # E8 NTU60 种子 1/2 + scratch 臂同步种子（E4 种子 1/2 已并入上方第 5 步）
  for s in 1 2; do
    python tools/pretrain_ntu.py --config configs/ntu60_cs.yaml --modality joint --seed $s >> "$QLOG" 2>&1 \
        && python tools/finetune_ntu.py --config configs/ntu60_cs.yaml --modality joint --seed $s >> "$QLOG" 2>&1 \
        && log "E8 seed$s ✅" || log "E8 seed$s 失败（可后补）"
    python tools/finetune_ntu.py --config configs/ntu60_cs.yaml --modality joint \
        --experiment ntu60_scratch --seed $s >> "$QLOG" 2>&1 \
        && log "E8c scratch seed$s ✅" || log "E8c scratch seed$s 失败（可后补）"
  done
fi

log "=== 队列完成。汇总: ==="
for f in runs/*/fusion_leakfree/fusion_report.txt runs/efficiency/efficiency_report.txt \
         runs/analysis/denoise_curve/*.csv; do
  [ -e "$f" ] && echo "  $f"
done
log "记得: ① 把 runs/ 全量回传本地 ② EXPERIMENTS.md §7 登记数字 ③ 释放实例前再查一遍"
