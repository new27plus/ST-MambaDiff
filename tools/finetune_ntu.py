import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import logging
import sys
import random

# --- 新增：src 路径引导 + 配置工具 ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.utils.config import (load_config, pick, make_run_dir, add_config_args,
                              resolve_experiment, set_seed, find_phase1_ckpt)

from feeders.ntu_feeder import NTU_Feeder
from models.har_classifier import HAR_Classifier_Model

def get_args():
    parser = argparse.ArgumentParser(description="Phase 2: NTU SOTA Fine-tuning (with Mixup)")
    # 数据集三项均可由 --config 提供；显式命令行优先
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--split', type=str, default=None, choices=['CS', 'CV', 'CSub', 'CSet'])
    parser.add_argument('--modality', type=str, default=None, choices=['joint', 'bone', 'motion', 'bone-motion'])
    parser.add_argument('--phase1_ckpt', type=str, default=None,
                        help='Phase1 权重路径（默认自动定位同 experiment/modality/seed 的最新产物）')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)      
    parser.add_argument('--weight_decay', type=float, default=0.0004) 
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    parser.add_argument('--alpha', type=float, default=0.2, help='Mixup alpha value')
    
    add_config_args(parser)
    return parser.parse_args()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class EarlyStopping:
    def __init__(self, patience=30, verbose=True, delta=0, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_score = -np.inf
        self.delta = delta
        self.mode = mode

    def __call__(self, val_score, model, path):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
            self.counter = 0

    def save_checkpoint(self, val_score, model, path):
        if self.verbose:
            print(f'Metric improved ({self.best_score:.4f} --> {val_score:.4f}). Saving model...')
        torch.save(model.state_dict(), path)

def setup_dirs(args):
    ckpt_dir = os.path.join(args.checkpoint_root, args.split, args.modality)
    log_dir = os.path.join(args.log_root, args.split, args.modality)
    os.makedirs(ckpt_dir, exist_ok=True) 
    return ckpt_dir, log_dir

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )

def get_scheduler(optimizer, num_epochs, warmup_epochs=10):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            t = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(t * np.pi))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    args = get_args()
    
    # --- 新增：配置驱动 ---
    cfg = load_config(args.config, args.overrides)
    experiment, seed = resolve_experiment(cfg, args, default_name='ntu_finetune')
    if args.data_path is None:
        args.data_path = pick(cfg, 'dataset.data_path')
    if args.split is None:
        args.split = pick(cfg, 'dataset.split')
    if args.modality is None:
        args.modality = pick(cfg, 'modality', 'joint')
    if args.data_path is None or args.split is None or args.modality is None:
        raise SystemExit('缺少 data_path/split/modality：请用 --config 提供或命令行显式指定')
    set_seed(seed)
    
    args.batch_size      = pick(cfg, 'phase2.batch_size', args.batch_size)
    args.lr              = pick(cfg, 'phase2.lr', args.lr)
    args.weight_decay    = pick(cfg, 'phase2.weight_decay', args.weight_decay)
    args.epochs          = pick(cfg, 'phase2.epochs', args.epochs)
    args.warmup_epochs   = pick(cfg, 'phase2.warmup_epochs', args.warmup_epochs)
    args.alpha           = pick(cfg, 'phase2.mixup_alpha', args.alpha)
    num_classes          = pick(cfg, 'dataset.num_classes', 120)
    num_joints           = pick(cfg, 'dataset.num_joints', 25)
    max_frames           = pick(cfg, 'dataset.max_frames', 64)
    num_workers          = pick(cfg, 'runtime.num_workers', 8)
    
    run_dir = make_run_dir(args.run_root, experiment, args.modality, seed, cfg, sys.argv)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)  # 【修复】漏建 logs/ 导致 FileHandler 打不开
    log_file = os.path.join(log_dir, 'phase2_mixup_train.log')
    setup_logging(log_file)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"==> Phase 2 SOTA (Mixup) Fine-tuning: {experiment} | split={args.split} | seed={seed}")
    logging.info(f"    Run dir: {run_dir}")
    
    # 1. 数据加载
    train_set = NTU_Feeder(args.data_path, split='train', window_size=max_frames, modality=args.modality)
    val_set = NTU_Feeder(args.data_path, split='test', window_size=max_frames, modality=args.modality)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    
    # 2. 模型初始化 (保持 128 通道)
    model = HAR_Classifier_Model(
        bone_pairs=[], 
        num_classes=num_classes,
        num_joints=num_joints,
        in_channels=3,
        max_frames=max_frames,
        model_channels=pick(cfg, 'model.model_channels', 128),  
        mamba_d_state=pick(cfg, 'model.mamba_d_state', 16),
        dropout=pick(cfg, 'model.dropout', 0.5),
        modality=args.modality,
        temporal_operator=pick(cfg, 'model.temporal_operator', 'auto')
    ).to(DEVICE)
    
    # 3. 加载 Phase 1 权重（自动定位同 experiment/modality/seed 的最新 phase1 产物）
    phase1_path = args.phase1_ckpt or find_phase1_ckpt(args.run_root, experiment, args.modality, seed)
    if phase1_path and os.path.exists(phase1_path):
        logging.info(f"Loading Phase 1 weights from: {phase1_path}")
        pretrained_dict = torch.load(phase1_path, map_location=DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('encoder.') and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        logging.warning(f"Phase 1 weights NOT found! Training from scratch.")

    # 4. 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = get_scheduler(optimizer, args.epochs, warmup_epochs=args.warmup_epochs)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=pick(cfg, 'phase2.label_smoothing', 0.1))
    early_stopping = EarlyStopping(patience=pick(cfg, 'phase2.patience', 40), verbose=True, mode='max')

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)
        for batch, label, _ in loop:
            data = batch.float().to(DEVICE)
            label = label.long().to(DEVICE)
            
            optimizer.zero_grad()
            
            # --- Mixup 逻辑 ---
            lam = np.random.beta(args.alpha, args.alpha)
            index = torch.randperm(data.size(0)).cuda()
            mixed_data = lam * data + (1 - lam) * data[index, :]
            logits = model(mixed_data)
            loss = lam * criterion(logits, label) + (1 - lam) * criterion(logits, label[index])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            preds = logits.argmax(1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch, label, _ in val_loader:
                data = batch.float().to(DEVICE)
                label = label.long().to(DEVICE)
                logits = model(data)
                preds = logits.argmax(1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
        
        val_acc = val_correct / val_total
        
        early_stopping(val_acc, model, os.path.join(ckpt_dir, 'phase2_sota_best.pt'))
            
        logging.info(f"Ep {epoch+1} | LR: {current_lr:.6f} | TrLoss: {avg_train_loss:.4f} | TrAcc(Mix): {train_acc:.4f} | ValAcc: {val_acc:.4f} (Best: {early_stopping.best_score:.4f})")
        
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    logging.info(f"Training Complete. Best Acc: {early_stopping.best_score:.4f}")

if __name__ == '__main__':
    main()