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

# 假设您的 Feeder 和模型代码位于以下路径
# 请确保您的目录结构与此匹配
from feeders.ucla_feeder import Feeder 
from models.st_unet_denoiser import ST_Unet_Denoising_Model

# ----------------------------------------------------------------------
# 1. 您必须修改的配置
# ----------------------------------------------------------------------
# --- 路径配置 ---
DATA_PATH = 'datasets/NW-UCLA/all_sqe'             # 包含 .json 骨架文件的根目录
TRAIN_LABEL_PATH = 'datasets/NW-UCLA/ucla_train_data.json' # 训练集标注文件
VAL_LABEL_PATH = 'datasets/NW-UCLA/ucla_val_data.json'   # 验证集标注文件
CHECKPOINT_PATH = './checkpoints'
LOG_DIR = './logs'
NUM_TRAIN_TIMESTEPS = 4000

def get_args():
    parser = argparse.ArgumentParser()
    # 默认值可以设为 'joint'，方便单独调试
    parser.add_argument('--modality', type=str, default='joint', 
                        choices=['joint', 'bone', 'motion', 'bone-motion'],
                        help='Modality to train')
    return parser.parse_args()

# 获取参数
args = get_args()
MODALITY = args.modality # 现在 MODALITY 变量来自命令行

print(f"\n>>> [Phase 1] Starting Pre-training for modality: {MODALITY} <<<\n")

BEST_LOSS_MODEL_NAME = f'phase1_{MODALITY}_best_loss.pt'
BEST_ACC_MODEL_NAME = f'phase1_{MODALITY}_best_acc.pt'
FINAL_MODEL_NAME = f'phase1_{MODALITY}final.pt'

# --- 训练配置 ---
NUM_EPOCHS = 300       # 【已修复】总训练轮数
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 数据集/模型 超参数 ---
# (这些必须与您的 Feeder 和模型定义匹配)
NUM_CLASSES = 10         # NW-UCLA有10个类别
NUM_JOINTS = 20
MAX_FRAMES = 52
IN_CHANNELS = 3
MODEL_CHANNELS = 128
MAMBA_D_STATE = 16
MODEL_CHANNELS_OUT = MODEL_CHANNELS * 4 # (C3 = 64 * 4 = 256)
PROBE_EPOCHS = 10           # 线性评估的快速训练轮数
PROBE_LR = 2e-4
PATIENCE = 30
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file_path):
    """配置日志：同时输出到文件和控制台"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除旧的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 1. 文件处理器 (写入日志文件)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # 2. 控制台处理器 (输出到屏幕)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

# 确保检查点目录存在
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 2. 依赖项 (您之前提供的 EarlyStopping 和骨架定义)
# ----------------------------------------------------------------------
class EarlyStopping:
    """
    (来自您之前提供的 utils/tools.py)
    """
    def __init__(self, patience=7, verbose=False, delta=0, mode='min', save_best_on_max=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.mode = mode # 'min' (for loss) or 'max' (for accuracy)
        self.save_best_on_max = save_best_on_max # Flag to save when score is max

    def __call__(self, val_metric, model, path):
        if self.mode == 'min':
            score = -val_metric
        else: # mode == 'max'
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, path)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, path):
        if self.verbose:
            if self.mode == 'min':
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_metric:.6f}).  Saving model ...')
            else: # mode == 'max'
                print(f'Validation metric increased ({self.val_loss_min:.6f} --> {val_metric:.6f}).  Saving model ...')
        
        # 保存整个 U-Net 模型
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_metric

# NW-UCLA 骨架 (0-based)
BONE_PAIRS_0_BASED = [
    (2-1, 1-1), (3-1, 2-1), (4-1, 3-1), (5-1, 3-1), (6-1, 5-1), (7-1, 6-1), 
    (8-1, 7-1), (9-1, 3-1), (10-1, 9-1), (11-1, 10-1), (12-1, 11-1), (13-1, 1-1), 
    (14-1, 13-1), (15-1, 14-1), (16-1, 15-1), (17-1, 1-1), (18-1, 17-1), 
    (19-1, 18-1), (20-1, 19-1)
]
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 3. 线性评估 (方法3) 的辅助模块
# ----------------------------------------------------------------------
class LinearProbeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LinearProbeClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 4. 主程序
# ----------------------------------------------------------------------
def main():
    # 1. 设置随机种子
    set_seed(SEED)

    # 2. 设置日志
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(LOG_DIR, f"pretrain_{MODALITY}_{timestamp}.txt")
    setup_logging(log_file_path)

    logging.info(f"Logging to: {log_file_path}")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Random Seed: {SEED}")
    logging.info(f"Modality: {MODALITY}")
    logging.info(f"Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # --- 初始化模型 ---
    logging.info("Initializing ST-Unet-Denoising-Model...")

    model = ST_Unet_Denoising_Model(
        bone_pairs=BONE_PAIRS_0_BASED,
        num_joints=NUM_JOINTS,
        in_channels=IN_CHANNELS,
        max_frames=MAX_FRAMES, 
        model_channels=MODEL_CHANNELS,
        mamba_d_state=MAMBA_D_STATE
    ).to(DEVICE)

    logging.info("Loading datasets...")
    train_dataset = Feeder(data_path=DATA_PATH, label_path=TRAIN_LABEL_PATH, modality=MODALITY, repeat=1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    val_dataset = Feeder(data_path=DATA_PATH, label_path=VAL_LABEL_PATH, modality=MODALITY, repeat=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    # --- 初始化训练组件 ---
    noise_scheduler = DDPMScheduler(NUM_TRAIN_TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 【已激活】初始化两个 EarlyStopping
    early_stopping_loss = EarlyStopping(patience=PATIENCE, verbose=True, mode='min')
    early_stopping_acc = EarlyStopping(patience=PATIENCE, verbose=True, mode='max')

    logging.info(f"\n>>> [Phase 1] Starting Pre-training for modality: {MODALITY} <<<\n")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- 2. 训练 ---
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        
        for step, (batch_data, labels, index) in enumerate(train_loop):
            clean_data = batch_data.squeeze(-1).to(DEVICE) 
            noise = torch.randn_like(clean_data)
            B = clean_data.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
            noisy_data = noise_scheduler.add_noise(clean_data, noise, timesteps)

            optimizer.zero_grad()
            predicted_noise = model(noisy_data, timesteps)
            loss = F.mse_loss(predicted_noise, noise)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)

        # --- 3. 验证 (方法 1: 监控去噪损失) ---
        model.eval()
        total_val_loss = 0
        val_loop_loss = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val Loss]", leave=False)
        
        with torch.no_grad():
            for batch_data, labels, index in val_loop_loss:
                clean_data = batch_data.squeeze(-1).to(DEVICE)
                noise = torch.randn_like(clean_data)
                B = clean_data.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
                noisy_data = noise_scheduler.add_noise(clean_data, noise, timesteps)
                
                predicted_noise = model(noisy_data, timesteps)
                val_loss = F.mse_loss(predicted_noise, noise)
                total_val_loss += val_loss.item()
            
        avg_val_loss = total_val_loss / len(val_loader)
        
        early_stopping_loss(avg_val_loss, model, os.path.join(CHECKPOINT_PATH, BEST_LOSS_MODEL_NAME))

        probe_classifier = LinearProbeClassifier(MODEL_CHANNELS_OUT, NUM_CLASSES).to(DEVICE)
        probe_optimizer = torch.optim.Adam(probe_classifier.parameters(), lr=PROBE_LR)
        probe_criterion = nn.CrossEntropyLoss()

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        probe_classifier.train()
        probe_train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Probe Train]", leave=False)
        for probe_epoch in range(PROBE_EPOCHS):
            for batch_data, labels, _ in probe_train_loop:
                clean_data = batch_data.squeeze(-1).to(DEVICE)
                labels = labels.long().to(DEVICE)
                
                dummy_t = torch.zeros(clean_data.shape[0], device=DEVICE).long()
                
                with torch.no_grad():
                    t_emb = model.time_embed(dummy_t)
                    h_mid, _ = model.encoder(clean_data, t_emb, model.A)
                
                logits = probe_classifier(h_mid)
                loss = probe_criterion(logits, labels)
                
                probe_optimizer.zero_grad()
                loss.backward()
                probe_optimizer.step()

        # 评估线性头
        probe_classifier.eval()
        total_correct = 0
        total_samples = 0
        probe_val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Probe Val]", leave=False)
        with torch.no_grad():
            for batch_data, labels, _ in probe_val_loop:
                clean_data = batch_data.squeeze(-1).to(DEVICE)
                labels = labels.long().to(DEVICE)
                dummy_t = torch.zeros(clean_data.shape[0], device=DEVICE).long()

                t_emb = model.time_embed(dummy_t)
                h_mid, _ = model.encoder(clean_data, t_emb, model.A)
                
                logits = probe_classifier(h_mid)
                _, preds = torch.max(logits, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        linear_probe_acc = total_correct / total_samples
        
        # 【已激活】调用 EarlyStopping (基于准确率)
        early_stopping_acc(linear_probe_acc, model, os.path.join(CHECKPOINT_PATH, BEST_ACC_MODEL_NAME))
        
        # 恢复主模型为 .train() 模式，为下一轮做准备
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        # --- 打印 Epoch 总结 (使用 logging) ---
        epoch_time = time.time() - start_time
        logging.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} Summary ---")
        logging.info(f"Time: {epoch_time:.2f}s")
        logging.info(f"Train Denoise Loss: {avg_train_loss:.6f}")
        logging.info(f"Val Denoise Loss:   {avg_val_loss:.6f} (Best: {early_stopping_loss.val_loss_min:.6f})")
        logging.info(f"Val Linear Probe Acc: {linear_probe_acc:.4f} (Best: {early_stopping_acc.val_loss_min:.4f})")
        logging.info("-" * 50)
        
        if early_stopping_loss.early_stop or early_stopping_acc.early_stop:
            logging.info("Early stopping triggered.")
            break

    # --- 5. 保存最终模型 ---
    logging.info("Saving final model...")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, FINAL_MODEL_NAME))
    logging.info(f"Phase 1 Pre-training complete. Best models saved in {CHECKPOINT_PATH}")

# ----------------------------------------------------------------------
# 6. 运行主函数
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()