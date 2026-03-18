import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import numpy as np
from tqdm import tqdm
import os
import time
import argparse
import logging
import sys
import random

from feeders.ntu_feeder import NTU_Feeder
from models.st_unet_denoiser import ST_Unet_Denoising_Model

def get_args():
    parser = argparse.ArgumentParser(description="Phase 1: NTU Pre-training (Physics-Consistent)")
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to .npz file')
    parser.add_argument('--split', type=str, required=True, choices=['CS', 'CV'], help='Split type')
    parser.add_argument('--modality', type=str, required=True, choices=['joint', 'bone', 'motion', 'bone-motion'])
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints_ntu')
    parser.add_argument('--log_root', type=str, default='./logs_ntu')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--epochs', type=int, default=80) 
    parser.add_argument('--patience', type=int, default=20)
    
    return parser.parse_args()

# --- 配置 ---
NUM_CLASSES = 60
NUM_JOINTS = 25
MAX_FRAMES = 64
MODEL_CHANNELS = 128    
MAMBA_D_STATE = 16      
NUM_WORKERS = 8       
NUM_TRAIN_TIMESTEPS = 1000 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BONE_PAIRS = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), 
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), 
    (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)
]

class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.inf if mode == 'min' else -np.inf
        self.delta = delta
        self.mode = mode

    def __call__(self, val_score, model, path):
        score = -val_score if self.mode == 'min' else val_score
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
            print(f'Metric improved. Saving model to {path}...')
        torch.save(model.state_dict(), path)

def setup_dirs(args):
    ckpt_dir = os.path.join(args.checkpoint_root, args.split, args.modality)
    log_dir = os.path.join(args.log_root, args.split, args.modality)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return ckpt_dir, log_dir

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )

class LinearProbeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.head(x)

def main():
    args = get_args()
    
    ckpt_dir, log_dir = setup_dirs(args)
    log_file = os.path.join(log_dir, 'phase1_train.log')
    setup_logging(log_file)
    
    logging.info(f"==> Phase 1 Pre-training: NTU-RGB+D 60 ({args.split})")
    logging.info(f"    Modality: {args.modality}")
    
    # 2. 数据加载
    # 【修改】使用 modality 字符串参数，window_size=64
    train_set = NTU_Feeder(args.data_path, split='train', window_size=MAX_FRAMES, modality=args.modality)
    val_set = NTU_Feeder(args.data_path, split='test', window_size=MAX_FRAMES, modality=args.modality)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    logging.info(f"    Train Samples: {len(train_set)} | Val Samples: {len(val_set)}")
    
    # 3. 模型初始化
    model = ST_Unet_Denoising_Model(
        bone_pairs=BONE_PAIRS,
        num_joints=NUM_JOINTS,
        in_channels=3,
        max_frames=MAX_FRAMES,
        model_channels=MODEL_CHANNELS,
        mamba_d_state=MAMBA_D_STATE,
        modality=args.modality # 【修改】传入模态，决定用 Mamba 还是 Conv
    ).to(DEVICE)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, mode='min')
    best_acc = 0.0 
    
    # 4. 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)
        for batch, _, _ in loop:
            clean_data = batch.float().to(DEVICE)
            noise = torch.randn_like(clean_data)
            t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (clean_data.size(0),), device=DEVICE).long()
            noisy_data = noise_scheduler.add_noise(clean_data, noise, t)
            
            optimizer.zero_grad()
            pred_noise = model(noisy_data, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Val Step ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, _, _ in val_loader:
                clean_data = batch.float().to(DEVICE)
                noise = torch.randn_like(clean_data)
                t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (clean_data.size(0),), device=DEVICE).long()
                noisy_data = noise_scheduler.add_noise(clean_data, noise, t)
                pred_noise = model(noisy_data, t)
                val_loss += F.mse_loss(pred_noise, noise).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # --- Linear Probe ---
        probe = LinearProbeClassifier(MODEL_CHANNELS*4, NUM_CLASSES).to(DEVICE)
        p_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        p_crit = nn.CrossEntropyLoss()
        
        probe.train()
        for i, (batch, label, _) in enumerate(train_loader):
            if i > 200: break 
            data = batch.float().to(DEVICE)
            label = label.long().to(DEVICE)
            with torch.no_grad():
                zero_t = torch.zeros(data.size(0), device=DEVICE).long()
                t_emb = model.time_embed(zero_t)
                h_mid, _ = model.encoder(data, t_emb, model.A)
            
            p_opt.zero_grad()
            p_crit(probe(h_mid), label).backward()
            p_opt.step()
            
        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, label, _ in val_loader:
                data = batch.float().to(DEVICE)
                label = label.long().to(DEVICE)
                zero_t = torch.zeros(data.size(0), device=DEVICE).long()
                t_emb = model.time_embed(zero_t)
                h_mid, _ = model.encoder(data, t_emb, model.A)
                preds = probe(h_mid).argmax(1)
                correct += (preds == label).sum().item()
                total += label.size(0)
        probe_acc = correct / total

        if probe_acc > best_acc:
            best_acc = probe_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'phase1_best_acc.pt'))

        early_stopping(avg_val_loss, model, os.path.join(ckpt_dir, 'phase1_best_loss.pt'))

        logging.info(f"Ep {epoch+1} | Time: {time.time()-start_time:.0f}s | "
                     f"TrLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | ProbeAcc: {probe_acc:.4f} (Best: {best_acc:.4f})")

        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'phase1_final.pt'))

if __name__ == '__main__':
    main()