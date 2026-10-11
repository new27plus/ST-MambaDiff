# -*- coding: utf-8 -*-
"""
E21: 测试时噪声鲁棒性曲线（可选加分项；pretrained vs scratch 退化对比）
====================================================================
对给定 phase2 分类器，在测试集输入上加固定生成器的加性高斯噪声
σ ∈ {0, 0.05, 0.1, 0.2, 0.4}，扫精度退化曲线。两个 checkpoint（预训练/scratch）
各跑一次 → 合并画图。噪声加在 feeder 输出之后、模型之前（测 encoder 鲁棒性，
与训练时增广无关；评估确定性：固定 torch.Generator 种子）。

用法（Experment 根目录）:
  python tools/noise_robustness.py --preset ntu60 \
      --ckpt artifacts/checkpoints/ntu60_cs/joint/phase2_sota_best.pt --label Pretrained
  python tools/noise_robustness.py --preset ntu60 \
      --ckpt "$(ls runs/ntu60_scratch/joint/seed0/*/checkpoints/phase2_sota_best.pt | head -1)" --label Scratch
输出: runs/analysis/robustness/robustness_{label}.csv（+全谱 PNG 由两 CSV 合并）
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = os.path.join(_ROOT, 'runs', 'analysis', 'robustness')

# 与 make_paper_figures.py 同款预设（避免循环导入，就地定义）
BONE_UCLA = [(1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7),
             (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
             (16, 15), (16, 0), (17, 16), (18, 17), (19, 18)]
BONE_NTU = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
            (23, 24), (24, 11)]
PRESETS = {
    'ucla':  dict(cfg='configs/ucla_final_arch.yaml', ntu=False, bone=BONE_UCLA),
    'ntu60': dict(cfg='configs/ntu60_cs.yaml', ntu=True, bone=BONE_NTU),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=list(PRESETS), default='ntu60')
    ap.add_argument('--ckpt', required=True, help='phase2 分类器权重')
    ap.add_argument('--label', required=True, help='曲线名，如 Pretrained / Scratch')
    ap.add_argument('--modality', default='joint')
    ap.add_argument('--sigmas', nargs='*', type=float, default=[0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from src.utils.config import load_config, pick
    from models.har_classifier import HAR_Classifier_Model
    from feeders.ucla_feeder import Feeder
    from feeders.ntu_feeder import NTU_Feeder

    pre = PRESETS[args.preset]
    cfg = load_config(os.path.join(_ROOT, pre['cfg']), args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HAR_Classifier_Model(
        bone_pairs=pre['bone'], num_classes=int(pick(cfg, 'dataset.num_classes', 60)),
        num_joints=int(pick(cfg, 'dataset.num_joints', 25)),
        in_channels=3, max_frames=int(pick(cfg, 'dataset.max_frames', 64)),
        model_channels=pick(cfg, 'model.model_channels', 128),
        mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
        dropout=0.0, modality=args.modality).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)

    if pre['ntu']:
        te = NTU_Feeder(pick(cfg, 'dataset.data_path'), split='test',
                        window_size=int(pick(cfg, 'dataset.max_frames', 64)), modality=args.modality)
    else:
        te = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                    label_path=pick(cfg, 'dataset.val_label_path'),
                    modality=args.modality, repeat=1)
    loader = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    rows = []
    for sigma in args.sigmas:
        g = torch.Generator(device='cpu'); g.manual_seed(args.seed)   # 每个 σ 同噪声序列（σ=0 除外）
        correct = total = 0
        with torch.no_grad():
            for batch, label, _ in loader:
                x = batch.float().to(device)
                if x.dim() == 5:
                    x = x.squeeze(-1)
                if sigma > 0:
                    noise = torch.randn(x.shape, generator=g).to(device)
                    x = x + sigma * noise
                pred = model(x).argmax(1).cpu().numpy()
                correct += (pred == label.numpy()).sum()
                total += len(label)
        acc = correct / total
        rows.append((sigma, acc))
        print(f'  σ={sigma:.2f}  acc={acc:.4f}  (n={total})')

    os.makedirs(OUT_DIR, exist_ok=True)
    csv = os.path.join(OUT_DIR, f'robustness_{args.label}.csv')
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('sigma,acc\n')
        for s, a in rows:
            f.write(f'{s},{a}\n')
    print(f'✅ {csv}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([r[0] for r in rows], [r[1] * 100 for r in rows], marker='o', label=args.label)
        ax.set_xlabel('input noise σ'); ax.set_ylabel('top-1 accuracy (%)')
        ax.set_title(f'Robustness to test-time noise ({args.preset})'); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(OUT_DIR, f'robustness_{args.label}.png')
        fig.savefig(png, dpi=200)
        print(f'✅ {png}（两 label 各跑一次后合并重画）')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
