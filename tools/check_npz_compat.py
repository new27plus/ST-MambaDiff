# -*- coding: utf-8 -*-
"""
NTU npz 兼容性检查器（五项，与 Experment/src/feeders/ntu_feeder.py 的契约对应）
用法:
    python tools/check_npz_compat.py --npz NTU60_CS.npz --classes 60
    python tools/check_npz_compat.py --npz NTU120_CSet.npz --classes 120
"""
import argparse
import numpy as np


def check(npz_path, num_classes):
    print(f"\n=== 检查 {npz_path} (期望类别数: {num_classes}) ===")
    ok = True
    try:
        npz = np.load(npz_path)
    except Exception as e:
        print(f"[FAIL] 无法打开文件: {e}")
        return False

    keys = set(npz.files)
    missing = {'x_train', 'y_train', 'x_test', 'y_test'} - keys
    if missing:
        print(f"[FAIL] 缺少 key: {missing}；实际: {sorted(keys)}")
        return False
    print(f"[OK]   keys 齐全: x_train / y_train / x_test / y_test")

    x_tr, y_tr = npz['x_train'], npz['y_train']
    x_te, y_te = npz['x_test'], npz['y_test']

    for name, x in [('x_train', x_tr), ('x_test', x_te)]:
        # 契约: feeder 读入后 reshape(N,T,2,25,3)，故存储形状应为 (N,T,150) 或 (N,T,2,25,3)
        if x.ndim == 3 and x.shape[2] == 150:
            print(f"[OK]   {name}: N={x.shape[0]}, T={x.shape[1]}, 形状=(N,T,150) → feeder reshape(N,T,2,25,3) ✓")
        elif x.ndim == 5 and x.shape[2:] == (2, 25, 3):
            print(f"[OK]   {name}: N={x.shape[0]}, T={x.shape[1]}, 形状=(N,T,2,25,3) ✓")
        else:
            print(f"[FAIL] {name} 形状 = {x.shape}，期望 (N,T,150) 或 (N,T,2,25,3)")
            ok = False

    for name, y in [('y_train', y_tr), ('y_test', y_te)]:
        if y.ndim != 2 or y.shape[1] != num_classes:
            print(f"[FAIL] {name} 形状 = {y.shape}，期望 (N, {num_classes}) one-hot")
            ok = False
            continue
        bad = int(((y > 0).sum(axis=1) == 0).sum())
        if bad:
            print(f"[FAIL] {name} 有 {bad} 行全零（feeder 取类会出错）")
            ok = False
        else:
            print(f"[OK]   {name}: one-hot 宽度 {y.shape[1]} ✓，每行均有正标签")

    for name, x in [('x_train', x_tr), ('x_test', x_te)]:
        sample = np.asarray(x[:200], dtype=np.float32)
        if np.isnan(sample).any():
            print(f"[WARN] {name} 抽样前 200 条存在 NaN")

    print("\n--- 样本量（与官方协议核对） ---")
    print(f"       train={x_tr.shape[0]}, test={x_te.shape[0]}, 合计={x_tr.shape[0]+x_te.shape[0]}")
    print("       参考: NTU60 CS≈40320/16560, CV≈38964/17916, NTU120 合计≈114480")

    print(f"\n{'[PASS] 可直接用于 src/feeders/ntu_feeder.py' if ok else '[FAIL] 按上面 FAIL 项处理'}")
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--classes', type=int, default=60, choices=[60, 120])
    a = p.parse_args()
    check(a.npz, a.classes)
