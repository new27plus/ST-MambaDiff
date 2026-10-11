import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
import sys
import logging
import argparse
import random

from torch.optim.lr_scheduler import CosineAnnealingLR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.har_classifier import HAR_Classifier_Model

from feeders.ucla_feeder import Feeder 

from src.utils.config import (load_config, pick, make_run_dir, add_config_args,
                              resolve_experiment, find_phase1_ckpt)

# --- 路径配置（默认值；实际以 --config YAML / 命令行为准，在 main 中解析覆盖） ---
DATA_PATH = os.path.join(_ROOT, 'datasets', 'NW-UCLA', 'all_sqe')
TRAIN_LABEL_PATH = os.path.join(_ROOT, 'datasets', 'NW-UCLA', 'ucla_train_data.json')
VAL_LABEL_PATH = os.path.join(_ROOT, 'datasets', 'NW-UCLA', 'ucla_val_data.json')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='joint', 
                        choices=['joint', 'bone', 'motion', 'bone-motion'],
                        help='Modality to train')
    parser.add_argument('--log_file', type=str, default=None, 
                        help='Path to save log file')
    parser.add_argument('--phase1_ckpt', type=str, default=None,
                        help='Phase1 权重路径（默认自动定位同 experiment/modality/seed 的最新产物）')
    add_config_args(parser)
    return parser.parse_args()

args = get_args()
MODALITY = args.modality

print(f"\n>>> [Phase 2] Starting Fine-tuning for modality: {MODALITY} <<<\n")

# --- 新的保存路径 ---
FINETUNED_MODEL_NAME = f'phase2_{MODALITY}_best_classifier.pt'

# --- 训练配置 ---
NUM_EPOCHS = 300       
LEARNING_RATE = 1e-4     
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 50          

# --- 数据集/模型 超参数 (必须与阶段一匹配) ---
NUM_CLASSES = 10         
NUM_JOINTS = 20
MAX_FRAMES = 52
IN_CHANNELS = 3
MODEL_CHANNELS = 128
MAMBA_D_STATE = 16

# (Feeder.bone 0-based)
BONE_PAIRS_0_BASED = [
    (2-1, 1-1), (3-1, 2-1), (4-1, 3-1), (5-1, 3-1), (6-1, 5-1), (7-1, 6-1), 
    (8-1, 7-1), (9-1, 3-1), (10-1, 9-1), (11-1, 10-1), (12-1, 11-1), (13-1, 1-1), 
    (14-1, 13-1), (15-1, 14-1), (16-1, 15-1), (17-1, 1-1), (18-1, 17-1), 
    (19-1, 18-1), (20-1, 19-1)
]
# ----------------------------------------------------------------------

SEED = 42
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 【新增】设置随机种子的函数
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    # 下面两行是为了确保 cuDNN 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# 2. 日志设置
# ----------------------------------------------------------------------
def setup_logging(log_file_path):
    """配置日志记录，同时输出到控制台和文件"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除任何现有的处理器，以防万一
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建控制台处理器 (打印到 stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # 创建文件处理器 (写入文件)
    if log_file_path:
        # 【修复】获取目录路径
        log_dir = os.path.dirname(log_file_path)
        
        # 【修复】仅在目录路径非空时才尝试创建目录
        if log_dir: 
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

# ----------------------------------------------------------------------
# 3. EarlyStopping (已修改为使用 logging)
# ----------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, mode='min', save_path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.inf
        self.val_metric_max = -np.inf # 用于 'max' 模式
        self.delta = delta
        self.mode = mode
        self.save_path = save_path

    def __call__(self, val_metric, model):
        if self.mode == 'min':
            score = -val_metric
        else: # mode == 'max'
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            if self.mode == 'min':
                logging.info(f'Validation metric decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_min = val_metric
            else: # mode == 'max'
                logging.info(f'Validation metric increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_max = val_metric
        
        torch.save(model.state_dict(), self.save_path)

# ----------------------------------------------------------------------
# 4. 主程序 (Main)
# ----------------------------------------------------------------------
def main(args):

    # --- 0. 配置解析：--config / --set 覆盖模块级默认值；输出到 runs/<experiment>/... ---
    global DATA_PATH, TRAIN_LABEL_PATH, VAL_LABEL_PATH, CHECKPOINT_PATH, LOG_DIR
    global NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS, PATIENCE, SEED
    global NUM_CLASSES, NUM_JOINTS, MAX_FRAMES, MODEL_CHANNELS, MAMBA_D_STATE
    
    cfg = load_config(args.config, args.overrides)
    experiment, seed = resolve_experiment(cfg, args, default_name='ucla_finetune')
    SEED = seed
    
    DATA_PATH        = pick(cfg, 'dataset.data_path', DATA_PATH)
    TRAIN_LABEL_PATH = pick(cfg, 'dataset.train_label_path', TRAIN_LABEL_PATH)
    VAL_LABEL_PATH   = pick(cfg, 'dataset.val_label_path', VAL_LABEL_PATH)
    NUM_EPOCHS       = pick(cfg, 'phase2.epochs', NUM_EPOCHS)
    LEARNING_RATE    = pick(cfg, 'phase2.lr', LEARNING_RATE)
    BATCH_SIZE       = pick(cfg, 'phase2.batch_size', BATCH_SIZE)
    NUM_WORKERS      = pick(cfg, 'runtime.num_workers', NUM_WORKERS)
    PATIENCE         = pick(cfg, 'phase2.patience', PATIENCE)
    NUM_CLASSES      = pick(cfg, 'dataset.num_classes', NUM_CLASSES)
    NUM_JOINTS       = pick(cfg, 'dataset.num_joints', NUM_JOINTS)
    MAX_FRAMES       = pick(cfg, 'dataset.max_frames', MAX_FRAMES)
    MODEL_CHANNELS   = pick(cfg, 'model.model_channels', MODEL_CHANNELS)
    MAMBA_D_STATE    = pick(cfg, 'model.mamba_d_state', MAMBA_D_STATE)
    weight_decay     = pick(cfg, 'phase2.weight_decay', 0.0004)
    
    run_dir = make_run_dir(args.run_root, experiment, MODALITY, SEED, cfg, sys.argv)
    CHECKPOINT_PATH = os.path.join(run_dir, 'checkpoints')
    LOG_DIR = os.path.join(run_dir, 'logs')
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # --- 关键！预训练模型路径（自动定位同 experiment/modality/seed 的最新 phase1 产物） ---
    PRETRAINED_MODEL_PATH = args.phase1_ckpt or find_phase1_ckpt(
        args.run_root, experiment, MODALITY, SEED, filename=f'phase1_{MODALITY}_best_acc.pt')
    
    set_seed(SEED)

    # --- 日志设置 ---
    if args.log_file is None:
        # 1. 如果未指定，自动生成文件名
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_basename = f"log_finetune_{timestr}.txt"
    else:
        # 2. 如果已指定，使用用户提供的文件名
        log_basename = args.log_file

    # 3. 【新】将基本文件名与 LOG_DIR 结合，确保所有日志都在 logs/ 目录下
    log_filename = os.path.join(LOG_DIR, log_basename)
    
    setup_logging(log_filename)
    logging.info(f"Logging training run to: {log_filename}")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Random Seed set to: {SEED}")

    # --- 1. 初始化阶段二模型 (HAR_Classifier_Model) ---
    logging.info("Initializing HAR_Classifier_Model...")
    model = HAR_Classifier_Model(
        bone_pairs=BONE_PAIRS_0_BASED,
        num_classes=NUM_CLASSES,
        num_joints=NUM_JOINTS,
        in_channels=IN_CHANNELS,
        max_frames=MAX_FRAMES, 
        model_channels=MODEL_CHANNELS,
        mamba_d_state=MAMBA_D_STATE,
        modality=MODALITY, # 【修复】最终架构必须按流传入模态，否则全部走 joint 分支
        temporal_operator=pick(cfg, 'model.temporal_operator', 'auto')
    ).to(DEVICE)

    # --- 2. 【关键】加载预训练权重 ---
    logging.info(f"Loading pre-trained weights from {PRETRAINED_MODEL_PATH}")
    try:
        pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        model_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and k.startswith(('encoder.', 'time_embed.'))
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logging.info(f"Successfully loaded {len(filtered_dict)} matching layers from pre-trained model.")
        
    except FileNotFoundError:
        logging.warning(f"Warning: Pre-trained model not found at {PRETRAINED_MODEL_PATH}.")
        logging.warning("Training from scratch (random initialization).")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        logging.warning("Training from scratch (random initialization).")

    # --- 3. 初始化数据加载器 ---
    logging.info("Loading datasets...")
    train_dataset = Feeder(data_path=DATA_PATH, label_path=TRAIN_LABEL_PATH, modality=MODALITY, repeat=1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    
    val_dataset = Feeder(data_path=DATA_PATH, label_path=VAL_LABEL_PATH, modality=MODALITY, repeat=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    # 【修改 A：Label Smoothing】
    logging.info("Using Label Smoothing (0.1)")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 【修改 B：优化器与权重衰减】支持差分学习率（E4b：encoder 低 LR 保住预训练先验）
    enc_scale = float(pick(cfg, 'phase2.encoder_lr_scale', 1.0))
    if enc_scale != 1.0:
        enc_params = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
        oth_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
        logging.info(f"Differential LR: encoder lr = {LEARNING_RATE * enc_scale:.2e} (x{enc_scale}), "
                     f"head lr = {LEARNING_RATE:.2e}")
        optimizer = torch.optim.AdamW(
            [{'params': enc_params, 'lr': LEARNING_RATE * enc_scale},
             {'params': oth_params, 'lr': LEARNING_RATE}],
            lr=LEARNING_RATE, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    
    # 【修改 C：Cosine Annealing Scheduler】
    logging.info("Using CosineAnnealingLR Scheduler")
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    early_stopping = EarlyStopping(
        patience=PATIENCE, 
        verbose=True, 
        mode='max',
        save_path=os.path.join(CHECKPOINT_PATH, FINETUNED_MODEL_NAME)
    )

    # --- 5. 训练循环 ---
    logging.info("Starting Phase 2 Fine-tuning...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        
        for step, (batch_data, labels, index) in enumerate(train_loop):
            data = batch_data.squeeze(-1).to(DEVICE)
            labels = labels.long().to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            
            # 【保留】梯度裁剪 (防止 Attention 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)
            
            train_loop.set_postfix(loss=loss.item())
        
        # 【修改 D：每个 Epoch 结束后更新学习率】
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_samples

        # --- 验证 ---
        model.eval()
        total_val_loss = 0
        total_correct = 0
        total_samples = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        
        with torch.no_grad():
            for batch_data, labels, index in val_loop:
                data = batch_data.squeeze(-1).to(DEVICE)
                labels = labels.long().to(DEVICE)
                logits = model(data)
                val_loss = criterion(logits, labels)
                _, preds = torch.max(logits, 1)
                total_val_loss += val_loss.item()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_correct / total_samples
        epoch_time = time.time() - start_time
        
        # --- 日志 ---
        logging.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} Summary ---")
        logging.info(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        logging.info(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.4f}")
        logging.info(f"Val Loss:   {avg_val_loss:.6f} | Val Acc:   {avg_val_acc:.4f}")
        
        early_stopping(avg_val_acc, model)
        
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    logging.info(f"Phase 2 Fine-tuning complete. Best model saved at {os.path.join(CHECKPOINT_PATH, FINETUNED_MODEL_NAME)}")
    logging.info(f"Best validation accuracy achieved: {early_stopping.val_metric_max:.4f}")




# ----------------------------------------------------------------------
# 5. 启动器
# 训练代码：python phase2_finetune.py --modality joint --log_file "baseline_scheduler_seed42.txt"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main(args)
