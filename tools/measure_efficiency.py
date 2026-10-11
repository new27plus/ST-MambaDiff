# -*- coding: utf-8 -*-
"""
E11: 统一效率测量（回应 R1-W1 / R2-M5 —— 一个口径说清所有配置的成本）
====================================================================
测什么（全部用 src/models 的最终架构 + thop 前向统计）:
  A. Phase-2 单流分类器，三种数据集配置（NTU60/NTU120/UCLA）
  B. Phase-1 完整 denoiser（同配置）
  C. N 流融合系统总成本（单流 × 流数 + 元分类器可忽略）
  D. 单流 FPS（当前设备，batch=1 与 batch=64）

用法（Experment 根目录）:
    python tools/measure_efficiency.py
依赖: pip install thop（缺了脚本会提示）
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
from models.har_classifier import HAR_Classifier_Model          # noqa: E402
from models.st_unet_denoiser import ST_Unet_Denoising_Model     # noqa: E402

try:
    from thop import profile
except ImportError:
    raise SystemExit('缺少 thop：pip install thop 后重试')

BONE_UCLA = [(1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7),
             (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
             (16, 15), (16, 0), (17, 16), (18, 17), (19, 18)]
BONE_NTU = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
            (23, 24), (24, 11)]

PRESETS = {  # name: (num_classes, num_joints, max_frames, bone_pairs, streams)
    'ntu60':  (60, 25, 64, BONE_NTU, 3),
    'ntu120': (120, 25, 64, BONE_NTU, 3),
    'ucla':   (10, 20, 52, BONE_UCLA, 4),
}


def measure(model, inp):
    """inp: 单个张量或参数元组。注意 thop 的 inputs 本身就是参数元组，
    不能再包一层（旧版 (inp,) 把 (x,t) 整体当成第一个参数导致 missing 't'）。"""
    model.eval()
    args = inp if isinstance(inp, tuple) else (inp,)
    flops, params = profile(model, inputs=args, verbose=False)
    return params / 1e6, flops / 1e9


def fps(model, inp, iters=50, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(*inp) if isinstance(inp, tuple) else model(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(*inp) if isinstance(inp, tuple) else model(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return iters / (time.time() - t0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(_ROOT, 'runs', 'efficiency')
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, 'efficiency_report.txt')
    lines = []

    def log(msg=''):
        lines.append(str(msg))
        print(msg, flush=True)

    log(f'=== E11 统一效率测量 | device={device} | temporal_operator=auto(论文架构) ===')
    for name, (nc, nj, tf, bp, n_streams) in PRESETS.items():
        log(f'\n--- 配置 {name} (classes={nc}, joints={nj}, frames={tf}) ---')
        # A. 单流分类器
        clf = HAR_Classifier_Model(bone_pairs=bp, num_classes=nc, num_joints=nj,
                                   in_channels=3, max_frames=tf, model_channels=128,
                                   mamba_d_state=16, dropout=0.0,
                                   modality='joint').to(device).eval()
        x = torch.randn(1, 3, tf, nj, device=device)
        p, g = measure(clf, x)
        log(f'  [A] 单流分类器: {p:.2f} M | {g:.3f} GFLOPs | FPS(b1)={fps(clf, x):.1f}')
        xb = torch.randn(64, 3, tf, nj, device=device)
        log(f'      FPS(b64)={fps(clf, xb):.1f}（≈吞吐 {fps(clf, xb)*64:.0f} 样本/秒）')
        # B. denoiser 全模型
        den = ST_Unet_Denoising_Model(bone_pairs=bp, num_joints=nj, in_channels=3,
                                      max_frames=tf, model_channels=128,
                                      mamba_d_state=16, modality='joint').to(device).eval()
        xd = torch.randn(1, 3, tf, nj, device=device)
        t = torch.zeros(1, dtype=torch.long, device=device)
        pd_, gd = measure(den, (xd, t))
        log(f'  [B] Phase-1 denoiser 全模型: {pd_:.2f} M | {gd:.3f} GFLOPs')
        # C. N 流融合系统
        log(f'  [C] {n_streams} 流融合系统: {p*n_streams:.2f} M | {g*n_streams:.3f} GFLOPs'
            f'（元分类器可忽略）')
        del clf, den
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    log('\n--- 论文用数建议 ---')
    log('  Table 1/3 的成本列用 [A] 对应配置的数；融合行用 [C]；')
    log('  Phase-1 预训练成本如需报告用 [B]，并在 4.1 注明为训练期一次性成本。')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n报告: {report}')


if __name__ == '__main__':
    main()
