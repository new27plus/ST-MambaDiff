import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import logging
import argparse
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

# 导入原来的模块
from models.har_classifier import HAR_Classifier_Model
from feeders.ucla_feeder import Feeder 

# ==========================================
# ⚙️ 配置 (保持与 SOTA 一致，除了预训练加载)
# ==========================================
# 注意路径要加 ../ 因为我们在 ablation 子目录里
DATA_PATH = '../datasets/NW-UCLA/all_sqe'
TRAIN_LABEL_PATH = '../datasets/NW-UCLA/ucla_train_data.json'
VAL_LABEL_PATH = '../datasets/NW-UCLA/ucla_val_data.json'
CHECKPOINT_PATH = './ablation_checkpoints' # 保存到新位置
LOG_DIR = './ablation_logs'

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# SOTA 参数
NUM_EPOCHS = 300       
LEARNING_RATE = 1e-4     
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 50           

# 模型参数 (SOTA 配置)
NUM_CLASSES = 10         
NUM_JOINTS = 20
MAX_FRAMES = 52
IN_CHANNELS = 3
MODEL_CHANNELS = 128    # SOTA Config
MAMBA_D_STATE = 16      # SOTA Config

BONE_PAIRS_0_BASED = [
    (2-1, 1-1), (3-1, 2-1), (4-1, 3-1), (5-1, 3-1), (6-1, 5-1), (7-1, 6-1), 
    (8-1, 7-1), (9-1, 3-1), (10-1, 9-1), (11-1, 10-1), (12-1, 11-1), (13-1, 1-1), 
    (14-1, 13-1), (15-1, 14-1), (16-1, 15-1), (17-1, 1-1), (18-1, 17-1), 
    (19-1, 18-1), (20-1, 19-1)
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='joint', choices=['joint', 'bone', 'motion', 'bone-motion'])
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = get_args()
    modality = args.modality
    set_seed(42)
    
    # 日志文件
    log_file = os.path.join(LOG_DIR, f'no_pretrain_{modality}.log')
    
    # 简单的日志配置
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], format='%(message)s')
    
    logging.info(f">>> ABLATION STUDY: Training {modality} FROM SCRATCH (No Pre-training) <<<")

    # 1. 初始化模型
    model = HAR_Classifier_Model(
        bone_pairs=BONE_PAIRS_0_BASED,
        num_classes=NUM_CLASSES,
        num_joints=NUM_JOINTS,
        in_channels=IN_CHANNELS,
        max_frames=MAX_FRAMES, 
        model_channels=MODEL_CHANNELS,
        mamba_d_state=MAMBA_D_STATE
    ).to(DEVICE)

    # -------------------------------------------------------------
    # ❌ ABLATION CHANGE: 不加载预训练权重
    # -------------------------------------------------------------
    logging.info("!!! NOTICE: Skipping pre-trained weights loading. Initializing with random weights.")
    # -------------------------------------------------------------

    # 2. 数据集
    train_dataset = Feeder(data_path=DATA_PATH, label_path=TRAIN_LABEL_PATH, modality=modality, repeat=1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    val_dataset = Feeder(data_path=DATA_PATH, label_path=VAL_LABEL_PATH, modality=modality, repeat=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # SOTA 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0004)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 只在第一个 epoch 和每 10 个 epoch 打印进度条，减少日志体积
        loop = tqdm(train_loader, leave=False)
        
        for data, labels, _ in loop:
            data = data.squeeze(-1).to(DEVICE)
            labels = labels.long().to(DEVICE)
            
            optimizer.zero_grad()
            # 注意: t_emb=None 已在模型内部处理
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data = data.squeeze(-1).to(DEVICE)
                labels = labels.long().to(DEVICE)
                logits = model(data)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            logging.info(f"Epoch {epoch+1}: New Best Val Acc: {val_acc:.4f}")
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'no_pretrain_{modality}_best.pt'))
        
        if (epoch + 1) % 10 == 0:
             logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    logging.info(f"Final Best Accuracy for {modality} (No Pre-train): {best_acc:.4f}")

if __name__ == "__main__":
    main()