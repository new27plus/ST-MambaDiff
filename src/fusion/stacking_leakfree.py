# -*- coding: utf-8 -*-
"""
无泄露版多流 Stacking 融合（修复审稿人1-W2指出的数据泄露）
====================================================================
旧协议（已封存于 artifacts/legacy_code/ucla_jan2026/stacking_fusion.py）的问题：
    仅在测试集上提取 logits，并直接在测试集内部做 5 折交叉验证 ——
    元分类器由 80% 的测试样本训练，属数据泄露，96.90% 数字不可用。

本脚本的正确协议（三步）：
    1. 对每个流，分别在 训练集 与 测试集 上前向提取 logits；
    2. 元分类器(LogisticRegression) 只用 训练集 学习：
       a) 训练集内 StratifiedKFold 生成 out-of-fold 预测（用于融合质量的
          无偏估计，即 sklearn stacking 的标准做法）；
       b) 再用全部训练集 logits 拟合最终元分类器；
    3. 测试集 logits 只前向一次得到最终数字 —— 这才是可写进论文的融合精度。

输出（保存到 run_dir）：
    - 各流单流精度（train / test）
    - OOF 融合精度（训练侧无偏估计）
    - 测试集融合精度（论文口径）
    - 旧协议(测试集内CV)的对照数字（仅用于向审稿人展示泄露影响，不进论文）
    - 各流贡献度（LR 系数绝对值归一）

用法：
    python src/fusion/stacking_leakfree.py \
        --config configs/ucla_final_arch.yaml \
        --experiment ucla_final --seed 0
（依赖 runs/<experiment>/<modality>/seed<k>/<ts>/checkpoints/phase2_*_best_classifier.pt）
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config import (load_config, pick, add_config_args,
                              resolve_experiment, find_phase1_ckpt, set_seed)

import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from models.har_classifier import HAR_Classifier_Model
from feeders.ucla_feeder import Feeder

# NW-UCLA 骨架树（0-based，与 tools/finetune_ucla.py 一致）
BONE_PAIRS_0_BASED = [
    (2-1, 1-1), (3-1, 2-1), (4-1, 3-1), (5-1, 3-1), (6-1, 5-1), (7-1, 6-1),
    (8-1, 7-1), (9-1, 3-1), (10-1, 9-1), (11-1, 10-1), (12-1, 11-1), (13-1, 1-1),
    (14-1, 13-1), (15-1, 14-1), (16-1, 15-1), (17-1, 1-1), (18-1, 17-1),
    (19-1, 18-1), (20-1, 19-1)
]


def get_logits(model, loader, device):
    """对整个 loader 前向一次，返回 (logits[N,C], labels[N])。"""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch, label, _ in loader:
            data = batch.squeeze(-1).to(device)
            out = model(data)
            logits_list.append(out.cpu().numpy())
            labels_list.append(label.numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def main():
    parser = argparse.ArgumentParser(description='Leak-free multi-stream stacking fusion')
    parser.add_argument('--modality_list', nargs='*', default=None,
                        help='流列表（默认取 YAML 的 streams）')
    parser.add_argument('--batch_size', type=int, default=64)
    add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    experiment, seed = resolve_experiment(cfg, args, default_name='ucla_final')
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    streams = args.modality_list or pick(cfg, 'streams', ['joint', 'bone', 'motion', 'bone-motion'])
    num_classes = pick(cfg, 'dataset.num_classes', 10)
    num_joints = pick(cfg, 'dataset.num_joints', 20)
    max_frames = pick(cfg, 'dataset.max_frames', 52)
    data_path = pick(cfg, 'dataset.data_path')
    train_labels = pick(cfg, 'dataset.train_label_path')
    val_labels = pick(cfg, 'dataset.val_label_path')
    meta_C = pick(cfg, 'fusion.meta_C', 0.1)
    meta_iter = pick(cfg, 'fusion.meta_max_iter', 2000)
    n_splits = pick(cfg, 'fusion.oof_splits', 5)

    run_dir = os.path.join(args.run_root, experiment, 'fusion', f'seed{seed}')
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, 'fusion_report.txt')

    def log(msg):
        print(msg)
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

    log('=' * 60)
    log(f'[Leak-free Stacking] experiment={experiment} seed={seed}')
    log(f'  streams={streams}')
    log(f'  meta: LogisticRegression(C={meta_C}, max_iter={meta_iter}), '
        f'OOF {n_splits} folds (train only)')
    log('=' * 60)

    train_logits, test_logits, y_train, y_test = [], [], None, None

    for m in streams:
        ckpt = find_phase1_ckpt(args.run_root, experiment, m, seed,
                                filename=f'phase2_{m}_best_classifier.pt')
        if ckpt is None:
            raise SystemExit(f'未找到流 [{m}] 的 phase2 checkpoint，请先完成该流微调。'
                             f'（查找位置 runs/{experiment}/{m}/seed{seed}/*/checkpoints/）')
        log(f'\n[{m}] loading {ckpt}')
        model = HAR_Classifier_Model(
            bone_pairs=BONE_PAIRS_0_BASED,
            num_classes=num_classes,
            num_joints=num_joints,
            in_channels=3,
            max_frames=max_frames,
            model_channels=pick(cfg, 'model.model_channels', 128),
            mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
            modality=m,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        train_set = Feeder(data_path=data_path, label_path=train_labels, modality=m, repeat=1)
        val_set = Feeder(data_path=data_path, label_path=val_labels, modality=m, repeat=1)
        nw = pick(cfg, 'runtime.num_workers', 4)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                  num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=nw := pick(cfg, 'runtime.num_workers', 4),
                                  pin_memory=True, persistent_workers=nw > 0)

        tr_logits, tr_y = get_logits(model, train_loader, device)
        te_logits, te_y = get_logits(model, val_loader, device)
        train_logits.append(tr_logits)
        test_logits.append(te_logits)
        if y_train is None:
            y_train, y_test = tr_y, te_y
        log(f'  single-stream acc  train={accuracy_score(tr_y, tr_logits.argmax(1)):.4f} '
            f' test={accuracy_score(te_y, te_logits.argmax(1)):.4f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    X_train = np.concatenate(train_logits, axis=1)
    X_test = np.concatenate(test_logits, axis=1)
    meta_kwargs = dict(max_iter=meta_iter, C=meta_C, solver='lbfgs')

    # ---- 1) 训练集内 OOF（融合质量的无偏估计，训练集不接触测试集）----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros_like(y_train)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        clf = LogisticRegression(**meta_kwargs)
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        oof_pred[va_idx] = clf.predict(X_train[va_idx])
    oof_acc = accuracy_score(y_train, oof_pred)
    log(f'\n[1] Train OOF fusion accuracy (unbiased, train-only): {oof_acc:.4f}')

    # ---- 2) 全训练集拟合最终元分类器，测试集只前向一次（论文口径）----
    final_clf = LogisticRegression(**meta_kwargs)
    final_clf.fit(X_train, y_train)
    test_pred = final_clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    log(f'[2] Test fusion accuracy (paper-usable, single forward): {test_acc:.4f}')

    # ---- 3) 旧协议对照（测试集内 CV —— 泄露版，仅供向审稿人展示影响）----
    skf2 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    leaky_accs = []
    for tr_idx, te_idx in skf2.split(X_test, y_test):
        clf = LogisticRegression(**meta_kwargs)
        clf.fit(X_test[tr_idx], y_test[tr_idx])
        leaky_accs.append(accuracy_score(y_test[te_idx], clf.predict(X_test[te_idx])))
    log(f'[3] OLD leaky protocol (test-set internal CV, NOT for paper): '
        f'{np.mean(leaky_accs):.4f} +/- {np.std(leaky_accs):.4f}')

    # ---- 4) 流贡献度（最终元分类器系数）----
    n_cls = X_train.shape[1] // len(streams)
    coefs = np.abs(final_clf.coef_).reshape(num_classes, len(streams), n_cls)
    importance = coefs.sum(axis=(0, 2))
    importance = importance / importance.sum() * 100
    log('\n[4] Stream contribution (LR coef abs, normalized):')
    for m, w in zip(streams, importance):
        log(f'    {m:<12s}: {w:.2f}%')

    log(f'\nReport saved to: {report_path}')


if __name__ == '__main__':
    main()
