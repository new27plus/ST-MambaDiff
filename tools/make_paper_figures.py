# -*- coding: utf-8 -*-
"""
E20: 最终架构论文可视化三件套（替换旧架构时期的图，保证图文一致）
====================================================================
三个子命令：
  denoising   反向扩散过程图（替换正文 Fig.3 / 附录 4 图）
              —— 从纯噪声完整采样，记录指定 t 的骨架，输出时间步×帧网格图
  confusion   NTU60 混淆矩阵（替换 Fig.4/5）—— 用归档 phase2 权重推理测试集
  tsne        特征 t-SNE 两面板（新图，直接支撑 R2-M1/M2）—— pretrained vs scratch

示例（服务器或本地均可，Experment 根目录）:
  # 1) 去噪过程（UCLA joint denoiser，最终架构）
  python tools/make_paper_figures.py denoising \
      --ckpt "$(ls runs/ucla_final/joint/seed0/*/checkpoints/phase1_joint_best_acc.pt | head -1)" \
      --preset ucla --sample_idx 0
  # 2) NTU60 混淆矩阵（joint 流）
  python tools/make_paper_figures.py confusion \
      --ckpt artifacts/checkpoints/ntu60_cs/joint/phase2_sota_best.pt --preset ntu60
  # 3) t-SNE（NTU60: 预训练 vs scratch）
  python tools/make_paper_figures.py tsne --preset ntu60 \
      --ckpt artifacts/checkpoints/ntu60_cs/joint/phase1_best_acc.pt --label Pretrained
  python tools/make_paper_figures.py tsne --preset ntu60 \
      --ckpt runs/ntu60_scratch/joint/seed0/*/checkpoints/phase2_sota_best.pt --label Scratch
输出统一落 runs/analysis/figures/。
"""
import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config import load_config, pick   # 【修复】遗漏的导入

OUT_DIR = os.path.join(_ROOT, 'runs', 'analysis', 'figures')

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


def _datasets(cfg, pre, modality):
    """返回 (train_set, test_set)。"""
    from feeders.ucla_feeder import Feeder
    from feeders.ntu_feeder import NTU_Feeder
    if pre['ntu']:
        tf = int(pick(cfg, 'dataset.max_frames', 64))
        tr = NTU_Feeder(pick(cfg, 'dataset.data_path'), split='train', window_size=tf, modality=modality)
        te = NTU_Feeder(pick(cfg, 'dataset.data_path'), split='test', window_size=tf, modality=modality)
    else:
        tr = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                    label_path=pick(cfg, 'dataset.train_label_path'),
                    modality=modality, repeat=1)
        te = Feeder(data_path=pick(cfg, 'dataset.data_path'),
                    label_path=pick(cfg, 'dataset.val_label_path'),
                    modality=modality, repeat=1)
    return tr, te


def _load_denoiser(cfg, pre, ckpt, modality, device):
    from models.st_unet_denoiser import ST_Unet_Denoising_Model
    m = ST_Unet_Denoising_Model(
        bone_pairs=pre['bone'], num_joints=int(pick(cfg, 'dataset.num_joints', 25)),
        in_channels=3, max_frames=int(pick(cfg, 'dataset.max_frames', 64)),
        model_channels=pick(cfg, 'model.model_channels', 128),
        mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
        modality=modality).to(device).eval()
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    return m


def _load_classifier(cfg, pre, ckpt, modality, device):
    from models.har_classifier import HAR_Classifier_Model
    m = HAR_Classifier_Model(
        bone_pairs=pre['bone'], num_classes=int(pick(cfg, 'dataset.num_classes', 60)),
        num_joints=int(pick(cfg, 'dataset.num_joints', 25)),
        in_channels=3, max_frames=int(pick(cfg, 'dataset.max_frames', 64)),
        model_channels=pick(cfg, 'model.model_channels', 128),
        mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
        dropout=0.0, modality=modality).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd, strict=False)
    return m


# ----------------------------------------------------------------------
# 模式 1: 反向扩散过程
# ----------------------------------------------------------------------
def cmd_denoising(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from diffusers import DDPMScheduler

    pre = PRESETS[args.preset]
    cfg = load_config(os.path.join(_ROOT, pre['cfg']), args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    T = int(pick(cfg, 'phase1.num_train_timesteps', 1000))
    model = _load_denoiser(cfg, pre, args.ckpt, args.modality, device)
    scheduler = DDPMScheduler(num_train_timesteps=T)

    capture = sorted(set(args.timesteps), reverse=True)          # 如 999 750 500 250 100 0
    frames_per_t = args.frames                                    # 每个 t 展示的帧数
    _, te = _datasets(cfg, pre, args.modality)
    x0, label, _ = te[args.sample_idx]
    x0 = np.asarray(x0)
    if x0.ndim == 4 and x0.shape[-1] == 1:   # 【修复】UCLA feeder 样本带尾部单维 (3,52,20,1)
        x0 = x0[..., 0]
    x0 = torch.as_tensor(x0, dtype=torch.float32).unsqueeze(0).to(device)

    # 完整反向链：从纯噪声逐步到 x0，记录指定 t
    x = torch.randn_like(x0)
    snapshots = {}
    for t in range(T - 1, -1, -1):
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t_b)
        x = scheduler.step(eps, t, x).prev_sample
        if t in capture:
            snapshots[t] = x.clone().cpu().numpy()[0]            # (C,T,V)

    # 绘图: 行=timesteps，列=均匀取的 frames_per_t 帧，画 2D 骨架（x-y 投影）
    T_len = snapshots[capture[0]].shape[1]
    frame_ids = np.linspace(0, T_len - 1, frames_per_t).astype(int)
    fig, axes = plt.subplots(len(capture), frames_per_t, figsize=(2.0 * frames_per_t, 2.2 * len(capture)))
    axes = np.atleast_2d(axes)
    for r, t in enumerate(capture):
        sk = snapshots[t]                                          # (C,T,V)
        scale = np.abs(sk[:, :, :]).max() + 1e-9
        for c, fid in enumerate(frame_ids):
            ax = axes[r, c]
            j = sk[:, fid, :]                                      # (3,V)
            ax.scatter(j[0], j[1], s=12, c='tab:blue', zorder=2)
            for a, b in pre['bone']:
                ax.plot([j[0, a], j[0, b]], [j[1, a], j[1, b]], '-', lw=1, c='tab:blue', zorder=1)
            ax.set_xlim(-scale, scale); ax.set_ylim(-scale, scale)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'frame {fid}', fontsize=9)
            if c == 0:
                ax.set_ylabel(f't={t}', fontsize=10)
    fig.suptitle(f'Reverse diffusion ({args.preset}, sample #{args.sample_idx}, label={int(label)})', fontsize=12)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'denoising_{args.preset}_s{args.sample_idx}.png')
    fig.savefig(out, dpi=200); fig.savefig(out.replace('.png', '.pdf'))
    print(f'✅ {out}（+pdf）')


# ----------------------------------------------------------------------
# 模式 2: 混淆矩阵
# ----------------------------------------------------------------------
def cmd_confusion(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    pre = PRESETS[args.preset]
    cfg = load_config(os.path.join(_ROOT, pre['cfg']), args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_classifier(cfg, pre, args.ckpt, args.modality, device)
    _, te = _datasets(cfg, pre, args.modality)
    from torch.utils.data import DataLoader
    loader = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    preds, gts = [], []
    with torch.no_grad():
        for batch, label, _ in loader:
            x = batch.float().to(device)
            if x.dim() == 5:
                x = x.squeeze(-1)
            preds.append(model(x).argmax(1).cpu().numpy())
            gts.append(label.numpy())
    y_p, y_g = np.concatenate(preds), np.concatenate(gts)
    n_cls = int(pick(cfg, 'dataset.num_classes', 60))
    cm = confusion_matrix(y_g, y_p, labels=range(n_cls))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    acc = (y_p == y_g).mean()
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    ax.set_title(f'Confusion matrix ({args.preset}, {args.modality} stream) — acc {acc*100:.2f}%')
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'confusion_{args.preset}_{args.modality}.png')
    fig.tight_layout(); fig.savefig(out, dpi=200); fig.savefig(out.replace('.png', '.pdf'))
    np.save(os.path.join(OUT_DIR, f'confusion_{args.preset}_{args.modality}.npy'), cm)
    print(f'✅ {out}（+pdf+npy）| test acc = {acc*100:.2f}%')


# ----------------------------------------------------------------------
# 模式 3: t-SNE 两面板
# ----------------------------------------------------------------------
def cmd_tsne(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader

    pre = PRESETS[args.preset]
    cfg = load_config(os.path.join(_ROOT, pre['cfg']), args.overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    model = _load_classifier(cfg, pre, args.ckpt, args.modality, device)
    _, te = _datasets(cfg, pre, args.modality)
    loader = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    feats, labels, n = [], [], 0
    with torch.no_grad():
        for batch, label, _ in loader:
            x = batch.float().to(device)
            if x.dim() == 5:
                x = x.squeeze(-1)
            t_emb = model.time_embed(torch.zeros(x.size(0), device=device).long())
            h, _ = model.encoder(x, t_emb, model.A)
            feats.append(h.mean(dim=(2, 3)).flatten(1).cpu().numpy())
            labels.append(label.numpy())
            n += x.size(0)
            if n >= args.n_samples:
                break
    F = np.concatenate(feats)[: args.n_samples]
    y = np.concatenate(labels)[: args.n_samples]
    emb = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
               init='pca').fit_transform(F)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab20', s=10, alpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f't-SNE of frozen encoder features ({args.label}, {args.preset})')
    fig.colorbar(sc, ax=ax, label='action class')
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'tsne_{args.preset}_{args.label}.png')
    fig.tight_layout(); fig.savefig(out, dpi=200); fig.savefig(out.replace('.png', '.pdf'))
    np.save(os.path.join(OUT_DIR, f'tsne_{args.preset}_{args.label}_emb.npy'), emb)
    np.save(os.path.join(OUT_DIR, f'tsne_{args.preset}_{args.label}_y.npy'), y)
    print(f'✅ {out}（+pdf+npy；两面板跑两次不同 --label 后并排排版）')


def main():
    ap = argparse.ArgumentParser(description='E20 paper figures (final architecture)')
    sub = ap.add_subparsers(dest='mode', required=True)

    def common(p):
        p.add_argument('--preset', choices=list(PRESETS), default='ucla')
        p.add_argument('--ckpt', required=True)
        p.add_argument('--modality', default='joint')
        p.add_argument('--batch_size', type=int, default=32)
        p.add_argument('--num_workers', type=int, default=2)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--set', nargs='*', default=[], dest='overrides')

    p1 = sub.add_parser('denoising'); common(p1)
    p1.add_argument('--sample_idx', type=int, default=0)
    p1.add_argument('--timesteps', nargs='*', type=int, default=[999, 750, 500, 250, 100, 0])
    p1.add_argument('--frames', type=int, default=6)
    p2 = sub.add_parser('confusion'); common(p2)
    p3 = sub.add_parser('tsne'); common(p3)
    p3.add_argument('--label', default='Pretrained')
    p3.add_argument('--n_samples', type=int, default=1000)
    p3.add_argument('--perplexity', type=float, default=30)

    args = ap.parse_args()
    {'denoising': cmd_denoising, 'confusion': cmd_confusion, 'tsne': cmd_tsne}[args.mode](args)


if __name__ == '__main__':
    main()
