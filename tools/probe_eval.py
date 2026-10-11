# -*- coding: utf-8 -*-
"""
E12: 冻结线性探针（回应 R2-M1/M2 + §9 决策树的关键证据）
====================================================================
问题：扩散预训练到底学到了"可迁移的语义特征"，还是只是随机初始化的水平？
方法：冻结 encoder，只在其 h_mid 特征上训练一个 LogisticRegression，
      对比以下 encoder 的测试集探针精度：
        (a) ucla_final 的 phase1 权重（预训练 ND）
        (b) ucla_std_ddpm 的 phase1 权重（预训练标准 DDPM，跑了 E4 后可加）
        (c) 随机初始化（--random_init，即 scratch 的特征水平基线）
        (d) NTU60 预训练 encoder（规模对照，可选 --ckpt）

判读：
  预训练 >> 随机  → 特征有价值，是"微调洗掉了"（H1）→ 走 E4b 温和迁移
  预训练 ≈ 随机   → 预训练特征无可迁移判别性（H2）→ 考虑 E8c 规模效应 / 重定位

用法（Experment 根目录）:
    python tools/probe_eval.py --preset ucla --ckpt <phase1_best_acc.pt>
    python tools/probe_eval.py --preset ucla --random_init
    python tools/probe_eval.py --preset ntu60 --ckpt artifacts/checkpoints/ntu60_cs/joint/phase1_best_acc.pt
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
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.metrics import accuracy_score                      # noqa: E402

from src.utils.config import load_config, pick                  # noqa: E402

BONE_UCLA = [(1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7),
             (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
             (16, 15), (16, 0), (17, 16), (18, 17), (19, 18)]
BONE_NTU = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
            (23, 24), (24, 11)]

PRESETS = {
    'ucla':  dict(cfg='configs/ucla_final_arch.yaml', bone=BONE_UCLA, data_key='dataset.data_path'),
    'ntu60': dict(cfg='configs/ntu60_cs.yaml', bone=BONE_NTU, data_key='dataset.data_path'),
}


def extract_features(model, loader, device, max_samples=None):
    """冻结 encoder，取 h_mid 的 GAP 特征（与 phase1 探针同款接口）。"""
    feats, labels = [], []
    n = 0
    with torch.no_grad():
        for batch, label, _ in loader:
            x = batch.float().to(device)
            if x.dim() == 5:
                x = x.squeeze(-1)
            x = x if x.dim() == 4 else x
            zero_t = torch.zeros(x.size(0), device=device).long()
            t_emb = model.time_embed(zero_t)
            h_mid, _ = model.encoder(x, t_emb, model.A)
            f = h_mid.mean(dim=(2, 3)).flatten(1)          # (B, C) GAP
            feats.append(f.cpu().numpy())
            labels.append(label.numpy())
            n += x.size(0)
            if max_samples and n >= max_samples:
                break
    return np.concatenate(feats), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=list(PRESETS), default='ucla')
    ap.add_argument('--ckpt', type=str, default=None, help='phase1 denoiser 权重路径')
    ap.add_argument('--random_init', action='store_true', help='不加载权重（scratch 特征基线）')
    ap.add_argument('--modality', default='joint')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    add_cfg = ap
    add_cfg.add_argument('--config', default=None)
    add_cfg.add_argument('--set', nargs='*', default=[], dest='overrides')
    args = ap.parse_args()

    from src.utils.config import add_config_args  # noqa: F401  (已内联，保留一致性)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pre = PRESETS[args.preset]
    cfg_path = args.config or os.path.join(_ROOT, pre['cfg'])
    cfg = load_config(cfg_path, args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.st_unet_denoiser import ST_Unet_Denoising_Model
    from feeders.ucla_feeder import Feeder
    from feeders.ntu_feeder import NTU_Feeder

    if args.preset == 'ucla':
        bp, nj, tf = BONE_UCLA, pick(cfg, 'dataset.num_joints', 20), pick(cfg, 'dataset.max_frames', 52)
        tr_set = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                        label_path=pick(cfg, 'dataset.train_label_path'),
                        modality=args.modality, repeat=1)
        te_set = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                        label_path=pick(cfg, 'dataset.val_label_path'),
                        modality=args.modality, repeat=1)
        nw = 4
    else:
        bp, nj, tf = BONE_NTU, pick(cfg, 'dataset.num_joints', 25), pick(cfg, 'dataset.max_frames', 64)
        dp = pick(cfg, 'dataset.data_path')
        tr_set = NTU_Feeder(dp, split='train', window_size=tf, modality=args.modality)
        te_set = NTU_Feeder(dp, split='test', window_size=tf, modality=args.modality)
        nw = 8

    model = ST_Unet_Denoising_Model(
        bone_pairs=bp, num_joints=nj, in_channels=3, max_frames=tf,
        model_channels=pick(cfg, 'model.model_channels', 128),
        mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
        modality=args.modality).to(device).eval()

    if args.random_init:
        tag = 'RANDOM-INIT'
    else:
        assert args.ckpt, '需要 --ckpt 或 --random_init'
        sd = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        tag = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt))) + '/' + os.path.basename(args.ckpt)

    for p in model.parameters():
        p.requires_grad = False

    tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)
    te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    F_tr, y_tr = extract_features(model, tr_loader, device)
    F_te, y_te = extract_features(model, te_loader, device)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(F_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(F_te))

    out_dir = os.path.join(_ROOT, 'runs', 'analysis', 'probe')
    os.makedirs(out_dir, exist_ok=True)
    line = f'[{args.preset}] {tag} | modality={args.modality} | 冻结探针 test acc = {acc:.4f} ' \
           f'(train {F_tr.shape[0]}, test {F_te.shape[0]}, feat dim {F_tr.shape[1]})'
    print(line)
    with open(os.path.join(out_dir, 'probe_results.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print('判读参照: 随机约 0.10(10类)/0.008(120类)；预训练若与随机接近 → H2 成立')


if __name__ == '__main__':
    main()
