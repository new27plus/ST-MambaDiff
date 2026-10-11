# -*- coding: utf-8 -*-
"""
E3b: 旧架构 checkpoints + 无泄露融合（2×2 分解矩阵的最后一个空格）
====================================================================
目的：分离"协议水分"与"架构差异"。
    旧架构 + 泄露协议 = 96.98（已知，论文旧数字）
    旧架构 + 无泄露协议 = ?   ← 本脚本
    新架构 + 泄露协议 = 96.55/96.12（已测）
    新架构 + 无泄露协议 = 92.74±0.63（已测）

实现：加载封存的旧模型代码（artifacts/legacy_code/ucla_jan2026）+ 旧 pt
（artifacts/checkpoints/ucla），用与 src/fusion/stacking_leakfree.py 完全相同的
无泄露协议（训练集拟合 LR 元分类器，测试集一次前向）。
附带产出：旧 pt 单流精度的档案核对（应精确复现 93.53/91.16/88.79/88.58）。

用法（WSL，Experment 根目录）:
    python tools/fusion_legacy_leakfree.py
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_ROOT = os.path.join(_ROOT, 'artifacts', 'legacy_code', 'ucla_jan2026')
CKPT_DIR = os.path.join(_ROOT, 'artifacts', 'checkpoints', 'ucla')

# 旧代码必须在 src 之前入 path，且本脚本不得 import src.models（避免 models 包冲突）
sys.path.insert(0, LEGACY_ROOT)
sys.path.insert(0, _ROOT)

import torch                                                    # noqa: E402
from torch.utils.data import DataLoader                         # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.model_selection import StratifiedKFold             # noqa: E402
from sklearn.metrics import accuracy_score                      # noqa: E402

from feeders.ucla_feeder import Feeder                          # noqa: E402  (旧版 feeder)
from models.har_classifier import HAR_Classifier_Model          # noqa: E402  (旧版模型)

STREAMS = ['joint', 'bone', 'motion', 'bone-motion']
DATA_PATH = os.path.join(_ROOT, '..', 'Code', 'datasets', 'NW-UCLA', 'all_sqe')
TRAIN_LABELS = os.path.join(_ROOT, '..', 'Code', 'datasets', 'NW-UCLA', 'ucla_train_data.json')
VAL_LABELS = os.path.join(_ROOT, '..', 'Code', 'datasets', 'NW-UCLA', 'ucla_val_data.json')

# 与旧 stacking_fusion.py 完全一致的构造参数（保证 checkpoint 逐键加载）
MODEL_ARGS = dict(num_classes=10, num_joints=20, in_channels=3,
                  max_frames=52, model_channels=128, mamba_d_state=16)
BONE_PAIRS = [(0, 0)]


def get_logits(model, loader, device):
    model.eval()
    lo, la = [], []
    with torch.no_grad():
        for batch, label, _ in loader:
            out = model(batch.squeeze(-1).to(device))
            lo.append(out.cpu().numpy())
            la.append(label.numpy())
    return np.concatenate(lo), np.concatenate(la)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out', default=os.path.join(_ROOT, 'runs', 'legacy_eval'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = os.path.join(args.out, 'fusion_report.txt')

    def log(msg):
        print(msg, flush=True)
        with open(report, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('=== E3b: 旧架构 checkpoints + 无泄露 LR stacking（训练集拟合，测试集一次前向） ===')

    tr_parts, te_parts, y_tr, y_te = [], [], None, None
    for m in STREAMS:
        ck = os.path.join(CKPT_DIR, f'phase2_{m}_best_classifier.pt')
        model = HAR_Classifier_Model(bone_pairs=BONE_PAIRS, **MODEL_ARGS).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        tr_set = Feeder(data_path=DATA_PATH, label_path=TRAIN_LABELS, modality=m, repeat=1)
        te_set = Feeder(data_path=DATA_PATH, label_path=VAL_LABELS, modality=m, repeat=1)
        tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        tr_logits, tr_y = get_logits(model, tr_loader, device)
        te_logits, te_y = get_logits(model, te_loader, device)
        if y_tr is None:
            y_tr, y_te = tr_y, te_y
        tr_parts.append(tr_logits)
        te_parts.append(te_logits)
        log(f'[{m}] 单流（档案核对，应=93.53/91.16/88.79/88.58）: '
            f'train={accuracy_score(tr_y, tr_logits.argmax(1)):.4f} '
            f'test={accuracy_score(te_y, te_logits.argmax(1)):.4f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    X_tr, X_te = np.concatenate(tr_parts, 1), np.concatenate(te_parts, 1)
    kw = dict(max_iter=2000, C=0.1, solver='lbfgs')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_tr)
    for tr_i, va_i in skf.split(X_tr, y_tr):
        clf = LogisticRegression(**kw)
        clf.fit(X_tr[tr_i], y_tr[tr_i])
        oof[va_i] = clf.predict(X_tr[va_i])
    log(f'\n[1] Train OOF: {accuracy_score(y_tr, oof):.4f}')

    final = LogisticRegression(**kw)
    final.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, final.predict(X_te))
    log(f'[2] Test fusion（旧架构 + 无泄露协议，论文口径候选）: {acc:.4f}')
    log('\n对照: 旧架构+泄露协议=96.98 | 新架构+泄露=96.55/96.12 | 新架构+无泄露=92.74±0.63 | scratch+无泄露=94.18')
    log(f'报告: {report}')


if __name__ == '__main__':
    main()
