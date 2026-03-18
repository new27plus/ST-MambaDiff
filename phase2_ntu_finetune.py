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

from feeders.ntu_feeder import NTU_Feeder
from models.har_classifier import HAR_Classifier_Model

def get_args():
    parser = argparse.ArgumentParser(description="Phase 2: NTU SOTA Fine-tuning (with Mixup)")
    parser.add_argument('--data_path', type=str, required=True)
    # parser.add_argument('--split', type=str, required=True, choices=['CS', 'CV'])
    parser.add_argument('--split', type=str, required=True, choices=['CSub', 'CSet'])
    parser.add_argument('--modality', type=str, required=True, choices=['joint', 'bone', 'motion', 'bone-motion'])
    # parser.add_argument('--checkpoint_root', type=str, default='./checkpoints_ntu')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints_ntu120')
    parser.add_argument('--log_root', type=str, default='./logs_ntu')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)      
    parser.add_argument('--weight_decay', type=float, default=0.0004) 
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    parser.add_argument('--alpha', type=float, default=0.2, help='Mixup alpha value')
    
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
    ckpt_dir, log_dir = setup_dirs(args)
    log_file = os.path.join(log_dir, 'phase2_mixup_train.log')
    setup_logging(log_file)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"==> Phase 2 SOTA (Mixup) Fine-tuning: NTU ({args.split}) - {args.modality}")
    
    # 1. 数据加载
    # 【修改】使用 modality 字符串参数，window_size=64
    train_set = NTU_Feeder(args.data_path, split='train', window_size=64, modality=args.modality)
    val_set = NTU_Feeder(args.data_path, split='test', window_size=64, modality=args.modality)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # 2. 模型初始化 (保持 128 通道)
    model = HAR_Classifier_Model(
        bone_pairs=[], 
        # num_classes=60,
        num_classes=120,
        num_joints=25,
        in_channels=3,
        max_frames=64,
        model_channels=128,  
        mamba_d_state=16,
        dropout=0.5,
        modality=args.modality # 【修改】传入模态
    ).to(DEVICE)
    
    # 3. 加载 Phase 1 权重
    phase1_path = os.path.join(ckpt_dir, 'phase1_best_acc.pt')
    if os.path.exists(phase1_path):
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
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    early_stopping = EarlyStopping(patience=40, verbose=True, mode='max')

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