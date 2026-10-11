# -*- coding: utf-8 -*-
"""
E15: Pareto 散点图（R2-M4 定位图）—— accuracy vs GFLOPs（NTU60-CS 协议）
====================================================================
数据两个来源：
  1. 本方法：E11 实测成本（单流 0.565G/1.66M；3 流融合 1.695G/4.97M）
     + E10 诚实融合精度（88.86）；单流点用归档 joint 84.89 作参照（虚线空心）。
  2. baseline：Table 1 各方法论文自报的 CS 精度与 GFLOPs（硬编码于下方 BASELINES）。

用法（Experment 根目录；本地即可）:
    python tools/make_pareto.py
输出: runs/analysis/figures/pareto_ntu60cs.png|pdf
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'runs', 'analysis', 'figures')

# method: (GFLOPs, acc%, marker样式组)
BASELINES = {  # NTU60-CS；数字取自原论文 Table 1（引用原文，不重测）
    'ST-GCN':        (16.2, 81.5),
    '2s-AGCN':       (37.4, 88.5),
    'Shift-GCN':     (2.5,  90.7),
    'MS-G3D':        (5.22, 91.5),
    'CTR-GCN':       (1.97, 92.4),
    'InfoGCN':       (1.84, 92.3),
    'BlockGCN':      (1.63, 93.1),
    'SkateFormer':   (3.62, 93.5),
}
OURS = {  # E11 实测 + E10 诚实数字
    'Ours (fusion, 3-stream)': (1.695, 88.86),
}
OURS_REF = {'Ours (single joint stream)': (0.565, 84.89)}  # 参照点（虚线空心）


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (g, a) in BASELINES.items():
        ax.scatter(g, a, c='tab:gray', s=60, zorder=2)
        ax.annotate(name, (g, a), textcoords='offset points', xytext=(6, 4), fontsize=8)
    for name, (g, a) in OURS.items():
        ax.scatter(g, a, c='tab:red', s=140, marker='*', zorder=3)
        ax.annotate(name, (g, a), textcoords='offset points', xytext=(8, -12),
                    fontsize=9, color='tab:red', fontweight='bold')
    for name, (g, a) in OURS_REF.items():
        ax.scatter(g, a, facecolors='none', edgecolors='tab:red', s=90, zorder=2, linestyle='--')
        ax.annotate(name, (g, a), textcoords='offset points', xytext=(6, 6),
                    fontsize=8, color='tab:red')
    ax.set_xlabel('GFLOPs (per sample, NTU60-CS protocol)')
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_title('Accuracy–efficiency trade-off (NTU RGB+D 60, Cross-Subject)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = os.path.join(_OUT, f'pareto_ntu60cs.{ext}')
        fig.savefig(out, dpi=200)
        print(f'✅ {out}')
    print('提示: 如需加入更多方法或改用 UCLA 协议，编辑本脚本顶部的 BASELINES/OURS 字典即可。')


if __name__ == '__main__':
    main()
