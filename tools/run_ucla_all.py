# -*- coding: utf-8 -*-
"""
UCLA 一键训练脚本：按流依次完成 phase1(预训练) + phase2(微调)
====================================================================
特性：
  1. 一条命令跑完 4 条流（joint/bone/motion/bone-motion），顺序执行；
  2. 断点续跑：已完成的阶段自动跳过（按 runs/<experiment>/<modality>/seed<k>/
    /*/checkpoints/ 下的产物判断），中断后重新运行同一命令即可继续；
  3. 某条流失败不影响后续流（phase1 失败会自动跳过该流 phase2），最后打印汇总；
  4. 日志与 .pt 权重由各工具自动保存于 runs/<experiment>/<modality>/seed<k>/<时间戳>/
     （config_snapshot.yaml + command.txt + checkpoints/ + logs/，与之前一致）；
  5. 主控日志另存 runs/<experiment>/runner_<时间戳>.log。

用法（WSL，在 Experment 目录下）:
    python tools/run_ucla_all.py                          # 默认: ucla_final, 4流, seed 0
    python tools/run_ucla_all.py --seeds 0 1 2            # 三种子连跑（挂机）
    python tools/run_ucla_all.py --modalities joint       # 只跑 joint 流
    python tools/run_ucla_all.py --phases finetune        # 只补 phase2
"""
import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.utils.config import load_config, pick, find_phase1_ckpt  # noqa: E402


def log(master_log_path, msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(master_log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def stage_done(run_root, experiment, modality, seed, filename):
    return find_phase1_ckpt(run_root, experiment, modality, seed, filename=filename) is not None


def run_stage(script, cfg_path, experiment, modality, seed, run_root):
    cmd = [sys.executable, os.path.join(_ROOT, 'tools', script),
           '--config', cfg_path, '--modality', modality,
           '--experiment', experiment, '--seed', str(seed), '--run_root', run_root]
    print(f'  $ {" ".join(cmd)}', flush=True)
    return subprocess.call(cmd, cwd=_ROOT)


def main():
    ap = argparse.ArgumentParser(description='UCLA all-modality two-phase training runner')
    ap.add_argument('--config', default=os.path.join(_ROOT, 'configs', 'ucla_final_arch.yaml'))
    ap.add_argument('--seeds', nargs='*', type=int, default=[0])
    ap.add_argument('--modalities', nargs='*', default=None,
                    help='默认取 YAML 的 streams（4 条流全跑）')
    ap.add_argument('--phases', choices=['both', 'pretrain', 'finetune'], default='both')
    ap.add_argument('--run_root', default=os.path.join(_ROOT, 'runs'))
    args = ap.parse_args()

    cfg = load_config(args.config)
    experiment = pick(cfg, 'experiment', 'ucla_final')
    modalities = args.modalities or pick(cfg, 'streams',
                                         ['joint', 'bone', 'motion', 'bone-motion'])
    # 预检数据文件，缺了直接退出（避免挂机空跑）
    for key in ('data_path', 'train_label_path', 'val_label_path'):
        p = pick(cfg, f'dataset.{key}')
        if not p or not os.path.exists(p):
            raise SystemExit(f'[预检失败] dataset.{key} 不存在: {p}')

    master_log = os.path.join(args.run_root, experiment,
                              f'runner_{time.strftime("%Y%m%d-%H%M%S")}.log')
    os.makedirs(os.path.dirname(master_log), exist_ok=True)
    log(master_log, f'=== UCLA 一键训练开始 | experiment={experiment} | '
                    f'modalities={modalities} | seeds={args.seeds} | phases={args.phases} ===')
    log(master_log, f'数据预检通过: {pick(cfg, "dataset.data_path")}')

    results = []
    for seed in args.seeds:
        for m in modalities:
            p1_name = f'phase1_{m}_best_acc.pt'
            p2_name = f'phase2_{m}_best_classifier.pt'

            # ---- Phase 1 预训练 ----
            if args.phases in ('both', 'pretrain'):
                if stage_done(args.run_root, experiment, m, seed, p1_name):
                    log(master_log, f'[seed{seed}][{m}] Phase1 已完成，跳过')
                else:
                    log(master_log, f'[seed{seed}][{m}] Phase1 预训练开始 ...')
                    rc = run_stage('pretrain_ucla.py', args.config, experiment, m, seed, args.run_root)
                    ok1 = stage_done(args.run_root, experiment, m, seed, p1_name)
                    log(master_log, f'[seed{seed}][{m}] Phase1 退出码={rc} '
                                    f'{"且产物已生成 ✅" if ok1 else "（无产物）❌"}')
                    results.append((seed, m, 'phase1', rc == 0 and ok1))
                    if not ok1:
                        log(master_log, f'[seed{seed}][{m}] Phase1 失败，跳过该流 Phase2')
                        results.append((seed, m, 'phase2', False))
                        continue
            else:
                if not stage_done(args.run_root, experiment, m, seed, p1_name):
                    raise SystemExit(f'[seed{seed}][{m}] Phase1 产物缺失，无法微调')

            # ---- Phase 2 微调 ----
            if args.phases in ('both', 'finetune'):
                if stage_done(args.run_root, experiment, m, seed, p2_name):
                    log(master_log, f'[seed{seed}][{m}] Phase2 已完成，跳过')
                    results.append((seed, m, 'phase2', True))
                else:
                    log(master_log, f'[seed{seed}][{m}] Phase2 微调开始 ...')
                    rc = run_stage('finetune_ucla.py', args.config, experiment, m, seed, args.run_root)
                    ok2 = stage_done(args.run_root, experiment, m, seed, p2_name)
                    log(master_log, f'[seed{seed}][{m}] Phase2 退出码={rc} '
                                    f'{"且产物已生成 ✅" if ok2 else "（无产物）❌"}')
                    results.append((seed, m, 'phase2', rc == 0 and ok2))

    # ---- 汇总 ----
    log(master_log, '=== 汇总 ===')
    n_fail = 0
    for seed, m, phase, ok in results:
        log(master_log, f'  seed{seed:<3d} {m:<12s} {phase:<8s} {"✅" if ok else "❌ 失败"}')
        n_fail += (not ok)
    log(master_log, f'=== 结束: {len(results) - n_fail}/{len(results)} 成功 ===')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
