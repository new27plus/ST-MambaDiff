import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ... (保留大部分导入，与上面相同) ...
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import argparse
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.har_classifier import HAR_Classifier_Model
from feeders.ucla_feeder import Feeder 

# ==========================================
# 🧱 定义旧的分类头 (Legacy Head)
# ==========================================
class LegacyHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='joint')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    modality = args.modality
    set_seed(42)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(LOG_DIR, f'old_head_{modality}.log'), level=logging.INFO, filemode='w')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    logging.info(f">>> ABLATION STUDY: {modality} with LEGACY HEAD (GAP only) <<<")

    # 1. 初始化 SOTA 模型
    model = HAR_Classifier_Model(
        bone_pairs=[], # 占位，不影响训练
        num_classes=10,
        model_channels=MODEL_CHANNELS,
        mamba_d_state=MAMBA_D_STATE
    ).to('cuda')

    # -------------------------------------------------------------
    # 🔨 ABLATION CHANGE: 强行替换分类头
    # -------------------------------------------------------------
    logging.info("!!! NOTICE: Replacing Advanced Head with Legacy GAP Head.")
    encoder_out_channels = MODEL_CHANNELS * 4
    model.classification_head = LegacyHead(encoder_out_channels, 10).to('cuda')
    # -------------------------------------------------------------
    
    phase1_path = f'../experiments/UCLA/checkpoints/phase1_{modality}_best_acc.pt'
    if os.path.exists(phase1_path):
        logging.info(f"Loading Phase 1 weights from {phase1_path}")
        pretrained_dict = torch.load(phase1_path, map_location=DEVICE)
        model_dict = model.state_dict()
        
        # 过滤掉 classification_head 的参数，只加载 encoder
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('encoder.') and k in model_dict}
        
        # 严格性检查
        if len(pretrained_dict) == 0:
            logging.error("❌ ERROR: No matching 'encoder.' keys found! Check checkpoint file.")
        else:
            model.load_state_dict(pretrained_dict, strict=False)
            logging.info(f"✅ Successfully loaded {len(pretrained_dict)} encoder layers.")
            
    else:
        logging.error(f"❌ CRITICAL WARNING: Phase 1 weights NOT FOUND at {phase1_path}!")
        logging.error("    Training will proceed from SCRATCH, which invalidates this ablation experiment.")
        sys.exit(1)

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
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'old_head_{modality}_best.pt'))
        
        if (epoch + 1) % 10 == 0:
             logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    logging.info(f"Final Best Accuracy for {modality} (Legacy Head): {best_acc:.4f}")

if __name__ == "__main__":
    main()