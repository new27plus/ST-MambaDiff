# -*- coding: utf-8 -*-
"""
NTU npz → 裸 .npy 成员抽取（修复本地 WSL OOM：np.load 对 zip 容器静默忽略 mmap_mode，
访问 x_train 即全量解压 11.4GB 进内存；裸 .npy 才支持真 memmap 惰性分页）

原理：savez 把每个数组按 .npy 格式原样存进 zip，因此"抽取"= 把 zip 成员字节流
原样拷出（64MB 分块，内存占用恒定），产物就是合法的 .npy，可与 npz 同目录共存。

用法（本地 WSL，Experment 根目录；每个文件 2–5 分钟，内存占用 <100MB）:
    python tools/split_npz_to_npy.py datasets/NTU120/ntu120/NTU120_CSub.npz
    python tools/split_npz_to_npy.py datasets/NTU120/ntu120/NTU120_CSet.npz
    python tools/split_npz_to_npy.py datasets/NTU60/ntu/NTU60_CS.npz      # 可选
产物: <stem>_x_train.npy / _y_train.npy / _x_test.npy / _y_test.npy
之后 NTU_Feeder 会自动优先使用这些 .npy（真 memmap），命令无需任何改动。
"""
import os
import shutil
import sys
import time
import zipfile

import numpy as np

KEYS = ('x_train', 'y_train', 'x_test', 'y_test')
CHUNK = 64 * 1024 * 1024  # 64MB


def split(npz_path):
    stem = npz_path[:-4] if npz_path.endswith('.npz') else npz_path
    print(f'=== {npz_path} ===')
    with zipfile.ZipFile(npz_path) as zf:
        names = {n: n for n in zf.namelist()}
        # savez 的成员名通常为 'x_train.npy'（可能带目录前缀），按 key 匹配
        for key in KEYS:
            member = next((n for n in names if n.split('/')[-1] == f'{key}.npy'), None)
            if member is None:
                print(f'  [跳过] zip 内无 {key}.npy')
                continue
            out = f'{stem}_{key}.npy'
            if os.path.exists(out):
                print(f'  [已存在] {out}')
                continue
            t0 = time.time()
            with zf.open(member) as src, open(out + '.tmp', 'wb') as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            os.replace(out + '.tmp', out)
            # 校验: 裸 npy 必须能真 memmap
            arr = np.load(out, mmap_mode='r')
            kind = type(arr).__name__
            print(f'  ✅ {out}  {arr.shape} {arr.dtype}  ({os.path.getsize(out)/1e9:.1f} GB, '
                  f'{time.time()-t0:.0f}s, type={kind})')
            assert kind == 'memmap', '裸 npy 未产生 memmap——异常，请检查'
    print('完成。\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        split(p)
