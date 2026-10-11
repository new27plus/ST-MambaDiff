# -*- coding: utf-8 -*-
"""
E13: 去噪 MSE–t 曲线（回应 R2-M1 —— 用数据说明 Neutral Decoder 的代价与收益）
====================================================================
对给定 denoiser checkpoint，在测试集上扫描噪声水平 t∈[0,T)，计算
MSE(预测噪声, 真噪声) 随 t 的曲线。两个 checkpoint 各一条曲线（如
Neutral Decoder vs 标准 DDPM），若 ND 在大 t 处重建更差、但下游精度更高，
即为"decoder 无法走捷径"的定量证据。

用法（Experment 根目录）:
    python tools/denoise_curve.py --preset ucla \
        --ckpt runs/ucla_final/joint/seed0/<ts>/checkpoints/phase1_joint_best_acc.pt --label ND
    python tools/denoise_curve.py --preset ucla \
        --ckpt runs/ucla_std_ddpm/joint/seed0/<ts>/checkpoints/phase1_joint_best_acc.pt --label StdDDPM
多条曲线写进同一份 CSV/PNG；报告落在 runs/analysis/denoise_curve/。
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
from torch.utils.data import DataLoader                         # noqa: E402
from diffusers import DDPMScheduler                             # noqa: E402

from src.utils.config import load_config, pick                  # noqa: E402

BONE_UCLA = [(1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7),
             (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
             (16, 15), (16, 0), (17, 16), (18, 17), (19, 18)]
BONE_NTU = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
            (23, 24), (24, 11)]

PRESETS = {
    'ucla':  dict(cfg='configs/ucla_final_arch.yaml', ntu=False),
    'ntu60': dict(cfg='configs/ntu60_cs.yaml', ntu=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=list(PRESETS), default='ucla')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--label', required=True, help='曲线名（如 ND / StdDDPM）')
    ap.add_argument('--modality', default='joint')
    ap.add_argument('--n_samples', type=int, default=256, help='测试集抽样条数')
    ap.add_argument('--n_t', type=int, default=20, help='t 网格点数')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--config', default=None)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    args = ap.parse_args()

    pre = PRESETS[args.preset]
    cfg = load_config(args.config or os.path.join(_ROOT, pre['cfg']), args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    from models.st_unet_denoiser import ST_Unet_Denoising_Model
    from feeders.ucla_feeder import Feeder
    from feeders.ntu_feeder import NTU_Feeder

    if pre['ntu']:
        nj, tf, bp = pick(cfg, 'dataset.num_joints', 25), pick(cfg, 'dataset.max_frames', 64), BONE_NTU
        te_set = NTU_Feeder(pick(cfg, 'dataset.data_path'), split='test',
                            window_size=tf, modality=args.modality)
        nw = 8
    else:
        nj, tf, bp = pick(cfg, 'dataset.num_joints', 20), pick(cfg, 'dataset.max_frames', 52), BONE_UCLA
        te_set = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                        label_path=pick(cfg, 'dataset.val_label_path'),
                        modality=args.modality, repeat=1)
        nw = 4

    model = ST_Unet_Denoising_Model(
        bone_pairs=bp, num_joints=nj, in_channels=3, max_frames=tf,
        model_channels=pick(cfg, 'model.model_channels', 128),
        mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
        modality=args.modality).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True), strict=True)

    T = pick(cfg, 'phase1.num_train_timesteps', 1000)
    scheduler = DDPMScheduler(num_train_timesteps=T)
    loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # 固定抽样与固定噪声（两曲线可比的关键）
    xs = []
    for batch, _, _ in loader:
        x = batch.float().to(device)
        if x.dim() == 5:
            x = x.squeeze(-1)
        xs.append(x)
        if sum(z.size(0) for z in xs) >= args.n_samples:
            break
    X = torch.cat(xs)[: args.n_samples]

    ts = np.linspace(0, T - 1, args.n_t).astype(int)
    curve = []
    for t_val in ts:
        t = torch.full((X.size(0),), int(t_val), device=device).long()
        noise = torch.randn_like(X)
        x_t = scheduler.add_noise(X, noise, t)
        with torch.no_grad():
            pred = model(x_t, t)
        mse = torch.nn.functional.mse_loss(pred, noise).item()
        curve.append((int(t_val), mse))
        print(f'  t={t_val:5d}  MSE={mse:.6f}')

    out_dir = os.path.join(_ROOT, 'runs', 'analysis', 'denoise_curve')
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f'curve_{args.label}.csv')
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('t,mse\n')
        for t_val, mse in curve:
            f.write(f'{t_val},{mse}\n')
    print(f'CSV: {csv}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([c[0] for c in curve], [c[1] for c in curve], marker='o', label=args.label)
        ax.set_xlabel('diffusion timestep t')
        ax.set_ylabel('MSE(pred noise, noise)')
        ax.set_title('Denoising difficulty across noise levels')
        ax.legend()
        fig.tight_layout()
        png = os.path.join(out_dir, f'curve_{args.label}.png')
        fig.savefig(png, dpi=200)
        print(f'PNG: {png}（同一 --label 重复跑会覆盖；不同 label 各一条，画图时请合并）')
    except ImportError:
        print('（未装 matplotlib，仅输出 CSV）')


if __name__ == '__main__':
    main()
