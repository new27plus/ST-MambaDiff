# -*- coding: utf-8 -*-
"""
E14: CKA 相似度 + ProbeAcc 曲线（回应 R2-M2 + §9 H1 检验）
====================================================================
(a) CKA：同一批测试样本上，phase1（预训练后）与 phase2（微调后）的 encoder
    特征线性 CKA。CKA 高 = 微调只是"适配"，预训练表征被保留（H1 不成立）；
    CKA 低 = 微调把预训练表征"洗掉"了（H1 成立）。
(b) 权重漂移：encoder 各层 L2 相对变化，辅助定位洗掉发生在哪几层。
(c) ProbeAcc 曲线：从 phase1 训练日志提取逐 epoch 线性探针精度（无需重跑）。

用法（Experment 根目录）:
    python tools/cka_analysis.py --preset ucla --modality joint \
        --phase1 runs/ucla_final/joint/seed0/<ts>/checkpoints/phase1_joint_best_acc.pt \
        --phase2 runs/ucla_final/joint/seed0/<ts>/checkpoints/phase2_joint_best_classifier.pt
可选: --log <pretrain日志> 额外画 ProbeAcc 曲线。
"""
import argparse
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
from torch.utils.data import DataLoader                         # noqa: E402

from src.utils.config import load_config, pick                  # noqa: E402
from models.st_unet_denoiser import ST_Unet_Denoising_Model     # noqa: E402
from models.har_classifier import HAR_Classifier_Model          # noqa: E402

BONE_UCLA = [(1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7),
             (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
             (16, 15), (16, 0), (17, 16), (18, 17), (19, 18)]
BONE_NTU = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
            (23, 24), (24, 11)]

PRESETS = {'ucla': dict(cfg='configs/ucla_final_arch.yaml', ntu=False),
           'ntu60': dict(cfg='configs/ntu60_cs.yaml', ntu=True)}


def linear_cka(X, Y):
    """X: (N,d1), Y: (N,d2) —— 线性 CKA。1=完全一致，0=无关。
    注意: 分子必须是 ||XᵀY||²_F（平方）。旧版漏了平方导致数值被人为压低数百倍。"""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro')
    yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(xy / (xx * yy + 1e-12))


def get_feats(encoder_fn, loader, device, n_max=512):
    feats = []
    n = 0
    with torch.no_grad():
        for batch, _, _ in loader:
            x = batch.float().to(device)
            if x.dim() == 5:
                x = x.squeeze(-1)
            feats.append(encoder_fn(x).cpu().numpy())
            n += x.size(0)
            if n >= n_max:
                break
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=list(PRESETS), default='ucla')
    ap.add_argument('--modality', default='joint')
    ap.add_argument('--phase1', required=True)
    ap.add_argument('--phase2', required=True)
    ap.add_argument('--log', default=None, help='phase1 训练日志（可选，画 ProbeAcc 曲线）')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--config', default=None)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    args = ap.parse_args()

    pre = PRESETS[args.preset]
    cfg = load_config(args.config or os.path.join(_ROOT, pre['cfg']), args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nj = pick(cfg, 'dataset.num_joints', 20 if not pre['ntu'] else 25)
    tf = pick(cfg, 'dataset.max_frames', 52 if not pre['ntu'] else 64)
    bp = BONE_NTU if pre['ntu'] else BONE_UCLA
    ch = pick(cfg, 'model.model_channels', 128)
    ds = pick(cfg, 'model.mamba_d_state', 16)

    from feeders.ucla_feeder import Feeder
    from feeders.ntu_feeder import NTU_Feeder
    if pre['ntu']:
        te_set = NTU_Feeder(pick(cfg, 'dataset.data_path'), split='test',
                            window_size=tf, modality=args.modality)
        nw = 8
    else:
        te_set = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                        label_path=pick(cfg, 'dataset.val_label_path'),
                        modality=args.modality, repeat=1)
        nw = 4
    loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # 同一个分类器骨架，分别灌入 phase1 的 encoder（来自 denoiser）与 phase2 的 encoder
    def build():
        return HAR_Classifier_Model(bone_pairs=bp, num_classes=pick(cfg, 'dataset.num_classes', 10),
                                    num_joints=nj, in_channels=3, max_frames=tf,
                                    model_channels=ch, mamba_d_state=ds,
                                    dropout=0.0, modality=args.modality).to(device).eval()

    m1, m2 = build(), build()
    sd1 = torch.load(args.phase1, map_location=device, weights_only=True)
    sd2 = torch.load(args.phase2, map_location=device, weights_only=True)
    m1.load_state_dict(sd1, strict=False)   # 只用其 encoder.*；time_embed/decoder 忽略
    m2.load_state_dict(sd2, strict=True)

    def enc_fn(m):
        def fn(x):
            zero_t = torch.zeros(x.size(0), device=device).long()
            t_emb = m.time_embed(zero_t)
            h, _ = m.encoder(x, t_emb, m.A)
            return h.mean(dim=(2, 3)).flatten(1)   # GAP 特征 (B, C)
        return fn

    F1 = get_feats(enc_fn(m1), loader, device)
    F2 = get_feats(enc_fn(m2), loader, device)
    cka = linear_cka(F1, F2)

    # 权重漂移（encoder 各参数张量的相对 L2）
    drift = []
    for k in sd2:
        if k.startswith('encoder.') and k in sd1 and sd1[k].shape == sd2[k].shape:
            a, b = sd1[k].float().cpu().numpy(), sd2[k].float().cpu().numpy()
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
            drift.append((k, rel))
    drift.sort(key=lambda kv: -kv[1])

    out_dir = os.path.join(_ROOT, 'runs', 'analysis', 'cka')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'{args.preset}_{args.modality}'
    with open(os.path.join(out_dir, f'cka_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'CKA(phase1 encoder, phase2 encoder) = {cka:.4f}  [{args.modality}]\n')
        f.write('判读: >0.9 微调基本保留预训练表征 | 0.5~0.9 部分洗掉 | <0.5 基本洗掉(H1成立)\n\n')
        f.write('权重相对漂移 Top-15:\n')
        for k, r in drift[:15]:
            f.write(f'  {r:.3f}  {k}\n')
    print(f'CKA = {cka:.4f}（>0.9 保留 / <0.5 洗掉）| 报告: {out_dir}/cka_{tag}.txt')

    # (c) ProbeAcc 曲线
    if args.log and os.path.exists(args.log):
        eps, accs = [], []
        for line in open(args.log, encoding='utf-8', errors='ignore'):
            m = re.search(r'Epoch (\d+)/\d+ Summary', line)
            if m:
                cur = int(m.group(1))
            ma = re.search(r'Val Linear Probe Acc: ([0-9.]+)', line)
            if m and ma:
                eps.append(cur)
                accs.append(float(ma.group(1)))
        if eps:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(eps, accs, marker='.', linewidth=1)
                ax.set_xlabel('pretrain epoch')
                ax.set_ylabel('frozen linear probe acc (val)')
                ax.set_title('ProbeAcc during diffusion pre-training')
                fig.tight_layout()
                png = os.path.join(out_dir, f'probe_curve_{tag}.png')
                fig.savefig(png, dpi=200)
                print(f'ProbeAcc 曲线: {png}（{len(eps)} epochs）')
            except ImportError:
                print('（未装 matplotlib，跳过画图；数据点已在上文日志中）')
        else:
            print('（日志中未找到 ProbeAcc 行——NTU 日志格式为 "ProbeAcc: x.xxxx"，可手动 grep）')


if __name__ == '__main__':
    main()
