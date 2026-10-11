# -*- coding: utf-8 -*-
"""
轻量实验配置工具（无 Hydra 依赖，兼容现有 argparse 脚本）
====================================================================
职责：
  1. 读取每实验一份的 YAML（configs/*.yaml，自包含、无分层继承）；
  2. 支持命令行覆盖：--set key.subkey=value （可多条，支持点路径）；
  3. 每次运行生成 runs/<experiment>/<modality>/seed<k>/<时间戳>/ 目录，
     自动写入 config_snapshot.yaml（生效配置快照）与 command.txt（完整命令行）；
  4. 提供 phase1 checkpoint 自动定位（phase2 复用同 experiment/modality/seed
     的最新 phase1 产物，杜绝跨实验串权重）；
  5. 统一随机种子设置。

设计约定：
  - YAML 中键为扁平点路径友好的嵌套结构，例如 phase1.epochs；
  - 覆盖优先级：命令行 --set > --config YAML > 各工具内默认值；
  - 工具脚本通过 pick(cfg, 'phase1.epochs', 默认值) 取值，未给配置时行为
    与历史脚本完全一致（渐进式迁移，不破坏旧用法）。
"""
import argparse
import os
import random
import time

import numpy as np
import yaml


# ----------------------------------------------------------------------
# 路径常量：项目根 = 本文件上三级（src/utils/config.py -> Experment/）
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RUN_ROOT = os.path.join(ROOT, 'runs')


def _parse_value(raw):
    """把命令行字符串值解析成 YAML 标量（int/float/bool/None/str/list）。"""
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _set_dotted(d, dotted_key, value):
    """按点路径写嵌套字典：a.b.c = v"""
    keys = dotted_key.split('.')
    cur = d
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _get_dotted(d, dotted_key, default=None):
    """按点路径读嵌套字典，缺任一级返回 default。"""
    cur = d
    for k in dotted_key.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# YAML 中这些键若为相对路径，自动以项目根（Experment/）解析为绝对路径
_PATH_KEYS = {'data_path', 'train_label_path', 'val_label_path',
              'phase1_ckpt', 'ckpt_root', 'out_dir'}


def _absolutize_paths(d):
    """递归把 _PATH_KEYS 中的相对路径锚定到项目根，保证与工作目录无关。"""
    for k, v in d.items():
        if isinstance(v, dict):
            _absolutize_paths(v)
        elif k in _PATH_KEYS and isinstance(v, str) and v and not os.path.isabs(v):
            d[k] = os.path.normpath(os.path.join(ROOT, v))  # 支持 ../Code 等跨目录相对路径
    return d


def load_config(config_path, overrides=()):
    """读取 YAML 并应用 'key.sub=value' 覆盖列表，返回最终生效的配置字典。"""
    cfg = {}
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    for item in overrides or ():
        if '=' not in item:
            raise ValueError(f"--set 参数格式应为 key.sub=value，收到: {item!r}")
        k, v = item.split('=', 1)
        _set_dotted(cfg, k.strip(), _parse_value(v.strip()))
    return _absolutize_paths(cfg)


def pick(cfg, dotted_key, default=None):
    """工具脚本取配置值的统一入口。"""
    return _get_dotted(cfg, dotted_key, default)


def make_run_dir(run_root, experiment, modality, seed, cfg=None, argv=None):
    """
    创建本次运行的输出目录并写入快照：
        runs/<experiment>/<modality>/seed<k>/<YYYYmmdd-HHMMSS>/
            config_snapshot.yaml   # 生效配置（含覆盖）
            command.txt            # 完整命令行，便于复现
    返回该目录路径。同名 experiment/modality/seed 的历史运行（不同时间戳）互不覆盖。
    """
    ts = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(run_root or DEFAULT_RUN_ROOT,
                           experiment, modality, f'seed{seed}', ts)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)  # 中央兜底
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    if cfg is not None:
        with open(os.path.join(run_dir, 'config_snapshot.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    if argv is not None:
        with open(os.path.join(run_dir, 'command.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(argv) + '\n')
    return run_dir


def find_phase1_ckpt(run_root, experiment, modality, seed,
                     filename='phase1_best_acc.pt'):
    """
    定位同一 experiment/modality/seed 下最新一次 phase1 运行的 checkpoint。
    返回路径或 None（phase2 据此决定加载还是 from scratch）。
    """
    base = os.path.join(run_root or DEFAULT_RUN_ROOT, experiment, modality, f'seed{seed}')
    if not os.path.isdir(base):
        return None
    candidates = []
    for ts_dir in os.listdir(base):
        ckpt = os.path.join(base, ts_dir, 'checkpoints', filename)
        if os.path.isfile(ckpt):
            candidates.append(ckpt)
    if not candidates:
        return None
    return sorted(candidates)[-1]  # 时间戳字典序即时间序


def set_seed(seed):
    """固定 random / numpy / torch 种子（含 cudnn 确定性）。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def add_config_args(parser: argparse.ArgumentParser):
    """给现有 argparse 注入统一配置参数（各工具脚本复用）。"""
    parser.add_argument('--config', type=str, default=None,
                        help='实验 YAML 路径（configs/*.yaml）')
    parser.add_argument('--set', nargs='*', default=[], dest='overrides',
                        help='配置覆盖，格式 key.sub=value，可多条')
    parser.add_argument('--experiment', type=str, default=None,
                        help='实验名（默认取 YAML 的 experiment 字段）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（默认取 YAML 的 seed 字段，再默认 42）')
    parser.add_argument('--run_root', type=str, default=DEFAULT_RUN_ROOT,
                        help=f'输出根目录（默认 {DEFAULT_RUN_ROOT}）')
    return parser


def resolve_experiment(cfg, args, default_name='unnamed'):
    """experiment 与 seed 的统一解析（命令行 > YAML > 默认）。"""
    experiment = args.experiment or pick(cfg, 'experiment', default_name)
    seed = args.seed if args.seed is not None else pick(cfg, 'seed', 42)
    return experiment, int(seed)
