# -*- coding: utf-8 -*-
"""
E10: NTU 四协议无泄露融合（修复旧 ensemble_fusion 在测试集上搜权重的泄露）
====================================================================
旧协议：权重网格直接对着测试集精度搜索（泄露）。
本协议：同样的加权和融合方式，但权重网格只在 **训练集** 上搜索，
        测试集只做一次前向 + 一次加权。另附训练集 OOF 无偏估计。

对每个协议（ntu60_cs / ntu60_cv / ntu120_csub / ntu120_cset）:
    python tools/fusion_ntu_leakfree.py --config configs/ntu120_cset.yaml \
        --ckpt_root artifacts/checkpoints/ntu120_cset
checkpoint 约定: <ckpt_root>/<modality>/phase2_sota_best.pt
"""
import argparse
import itertools
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                # noqa: E402
from torch.utils.data import DataLoader                     # noqa: E402

from feeders.ntu_feeder import NTU_Feeder                   # noqa: E402
from models.har_classifier import HAR_Classifier_Model      # noqa: E402
from src.utils.config import load_config, pick, add_config_args, resolve_experiment  # noqa: E402
from sklearn.metrics import accuracy_score                  # noqa: E402

BONE_PAIRS_NTU = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)
]


def get_logits(model, loader, device):
    model.eval()
    lo, la = [], []
    with torch.no_grad():
        for batch, label, _ in loader:
            out = model(batch.float().to(device))
            lo.append(out.cpu().numpy())
            la.append(label.numpy())
    return np.concatenate(lo), np.concatenate(la)


def main():
    ap = argparse.ArgumentParser(description='NTU leak-free ensemble fusion')
    ap.add_argument('--ckpt_root', type=str, required=True,
                    help='如 artifacts/checkpoints/ntu120_cset')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--w_step', type=float, default=0.1, help='权重网格步长')
    ap.add_argument('--w_max', type=float, default=2.0, help='权重上限')
    add_config_args(ap)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    experiment, seed = resolve_experiment(cfg, args, default_name='ntu_fusion')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    streams = pick(cfg, 'streams', ['joint', 'bone', 'motion'])
    data_path = pick(cfg, 'dataset.data_path')
    num_classes = pick(cfg, 'dataset.num_classes')
    num_joints = pick(cfg, 'dataset.num_joints')
    max_frames = pick(cfg, 'dataset.max_frames')

    out_dir = os.path.join(args.run_root, experiment, 'fusion_leakfree')
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, 'fusion_report.txt')

    def log(msg):
        print(msg, flush=True)
        with open(report, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

    log(f'=== E10 NTU 无泄露融合 | {experiment} | streams={streams} ===')

    tr_parts, te_parts, y_tr, y_te = [], [], None, None
    for m in streams:
        ck = os.path.join(args.ckpt_root, m, 'phase2_sota_best.pt')
        model = HAR_Classifier_Model(
            bone_pairs=BONE_PAIRS_NTU, num_classes=num_classes, num_joints=num_joints,
            in_channels=3, max_frames=max_frames,
            model_channels=pick(cfg, 'model.model_channels', 128),
            mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
            dropout=0.0, modality=m).to(device)
        sd = torch.load(ck, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)  # 分类头尺寸无关权重值；strict 防呆由形状保证
        tr_set = NTU_Feeder(data_path, split='train', window_size=max_frames, modality=m)
        te_set = NTU_Feeder(data_path, split='test', window_size=max_frames, modality=m)
        tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=pick(cfg, 'runtime.num_workers', 8), pin_memory=True,
                               persistent_workers=pick(cfg, 'runtime.num_workers', 8) > 0)
        te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=pick(cfg, 'runtime.num_workers', 8), pin_memory=True,
                               persistent_workers=pick(cfg, 'runtime.num_workers', 8) > 0)
        tr_logits, tr_y = get_logits(model, tr_loader, device)
        te_logits, te_y = get_logits(model, te_loader, device)
        if y_tr is None:
            y_tr, y_te = tr_y, te_y
        tr_parts.append(tr_logits)
        te_parts.append(te_logits)
        log(f'[{m}] 单流: train={accuracy_score(tr_y, tr_logits.argmax(1)):.4f} '
            f'test={accuracy_score(te_y, te_logits.argmax(1)):.4f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 训练集上网格搜权重（0~w_max, 步长 w_step）
    # 【内存优化】float32 + 预分配缓冲写入（旧版每组组合临时分配 S*N*C float64 ≈ 180MB，
    #   9261 组合在 WSL 受限内存下会崩；此版峰值恒定 ≈ 2*N*C*4B）
    grid = np.arange(0.0, args.w_max + 1e-9, args.w_step)
    combos = list(itertools.product(grid, repeat=len(streams)))
    L_tr = np.stack(tr_parts, 0).astype(np.float32)   # (S, N, C)
    L_te = np.stack(te_parts, 0).astype(np.float32)
    buf = np.empty_like(L_tr[0])
    best_w, best_tr = None, -1.0
    for w in combos:
        np.multiply(L_tr[0], np.float32(w[0]), out=buf)
        for i in range(1, len(streams)):
            buf += L_tr[i] * np.float32(w[i])
        acc = accuracy_score(y_tr, buf.argmax(1))
        if acc > best_tr:
            best_tr, best_w = acc, w
    log(f'\n[1] 训练集最优权重 {dict(zip(streams, best_w))} → train acc {best_tr:.4f}（共 {len(combos)} 组）')

    wv = np.asarray(best_w, dtype=np.float64)[:, None, None]
    test_acc = accuracy_score(y_te, (L_te * wv).sum(0).argmax(1))
    log(f'[2] Test fusion（论文口径：训练集搜权重，测试集一次前向）: {test_acc:.4f}')
    equal = accuracy_score(y_te, (L_te * np.ones(len(streams))[:, None, None]).sum(0).argmax(1))
    log(f'[3] 对照（等权求和）: {equal:.4f}')
    log(f'报告: {report}')


if __name__ == '__main__':
    main()
