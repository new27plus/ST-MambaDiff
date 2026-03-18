import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import argparse
import sys
from diffusers import DDPMScheduler

# --- 引入项目组件 ---
from feeders.ntu_feeder import NTU_Feeder
from models.st_unet_denoiser import ST_Unet_Denoising_Model
from models.har_classifier import HAR_Classifier_Model

# --- 全局配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './images'
os.makedirs(SAVE_DIR, exist_ok=True)

# NTU 骨骼连接定义 (用于画图连线)
# 注意：这里索引是 0-based
BONES = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)
]

def get_args():
    parser = argparse.ArgumentParser(description="Visualization Tools for NTU SOTA Model")
    parser.add_argument('--task', type=str, required=True, choices=['denoise', 'tsne', 'cm'], help='Choose task')
    
    # 数据相关
    parser.add_argument('--data_path', type=str, default='datasets/NTU60/ntu/NTU60_CS.npz')
    parser.add_argument('--split', type=str, default='test', help='Use test set for viz')
    parser.add_argument('--modality', type=str, default='joint')
    
    # 权重路径
    parser.add_argument('--phase1_ckpt', type=str, default='./experiments/NTU60/checkpoints/CS/joint/phase1_best_loss.pt')
    parser.add_argument('--phase2_ckpt', type=str, default='./experiments/NTU60/checkpoints/CV/joint/phase2_sota_best.pt')
    
    # 模型配置 (必须与训练一致)
    parser.add_argument('--num_classes', type=int, default=60)
    parser.add_argument('--num_joints', type=int, default=25)
    parser.add_argument('--max_frames', type=int, default=64)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--save_format', type=str, default='pdf', choices=['pdf', 'svg', 'eps', 'png'])
    parser.add_argument('--sample_idx', type=int, default=-1)
    parser.add_argument('--output_name', type=str, default='')
    
    return parser.parse_args()


def build_save_path(file_stem, save_format):
    return os.path.join(SAVE_DIR, f'{file_stem}.{save_format}')

# =====================================================================
# 工具函数：画单个骨架
# =====================================================================
def plot_skeleton_3d(ax, data, title=""):
    """
    data: (3, V) numpy array
    """
    x = data[0, :]
    y = data[2, :]
    z = data[1, :]
    
    ax.clear()
    
    # 1. 调整视角
    ax.view_init(elev=20., azim=-60) 

    # 2. 画骨骼连线
    for v1, v2 in BONES:
        ax.plot([x[v1], x[v2]], [y[v1], y[v2]], [z[v1], z[v2]], c='black', linewidth=1.5, alpha=0.7)
    
    # 3. 画关节
    ax.scatter(x, y, z, c='red', marker='o', s=20, alpha=0.9, edgecolors='white', linewidth=0.5)

    # 4. 强制坐标轴比例一致 (防止变形)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 5. 隐藏坐标轴刻度 (为了论文美观)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.axis('off') # 取消注释这一行可以去掉背景网格和坐标轴线，更加干净
    
    ax.set_title(title, fontsize=10, y=1.05)

# =====================================================================
# 任务 1: 扩散去噪过程可视化
# =====================================================================
def run_denoising_viz(args):
    print("🎨 Generating Denoising Process Visualization...")
    
    # 1. 加载模型
    model = ST_Unet_Denoising_Model(
        bone_pairs=[],
        num_joints=args.num_joints,
        in_channels=3,
        max_frames=args.max_frames,
        model_channels=args.model_channels,
        mamba_d_state=16,
        modality=args.modality
    ).to(DEVICE)
    
    if os.path.exists(args.phase1_ckpt):
        model.load_state_dict(torch.load(args.phase1_ckpt, map_location=DEVICE))
        print(f"   Loaded Phase 1 weights: {args.phase1_ckpt}")
    else:
        print(f"❌ Checkpoint not found: {args.phase1_ckpt}")
        return

    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 2. 获取一个真实样本
    dataset = NTU_Feeder(args.data_path, split=args.split, window_size=args.max_frames, modality=args.modality)
    if args.sample_idx >= 0:
        if args.sample_idx >= len(dataset):
            raise ValueError(f"sample_idx out of range: {args.sample_idx}, dataset size: {len(dataset)}")
        idx = args.sample_idx
    else:
        idx = np.random.randint(0, len(dataset))
    print(f"Current Sample Index: {idx}")
    real_sample, label, _ = dataset[idx] # (3, 64, 25)
    
    # 3. 逆向扩散过程 (Reverse Diffusion)
    # 生成纯噪声
    x_t = torch.randn(1, 3, args.max_frames, args.num_joints).to(DEVICE)
    
    # 我们要捕捉的时间点 (从大到小)
    capture_steps = [999, 750, 500, 250, 0]
    captured_samples = []

    print("   Running reverse diffusion process...")
    with torch.no_grad():
        for t in tqdm(reversed(range(0, 1000)), total=1000):
            # 模型预测噪声
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            pred_noise = model(x_t, t_tensor)
            
            # Scheduler 更新 x_t -> x_{t-1}
            # diffusers 的 step 返回的是一个对象，.prev_sample 是更新后的样本
            x_t = scheduler.step(pred_noise, t, x_t).prev_sample
            
            if t in capture_steps:
                captured_samples.append((t, x_t.cpu().numpy()[0]))

    # 4. 绘图
    fig = plt.figure(figsize=(20, 5))
    # 取中间一帧来画 (例如第 30 帧)
    frame_idx = 30 
    
    # 把真实样本也放进去对比
    captured_samples.append(("Ground Truth", real_sample))
    
    for i, (t_val, sample_data) in enumerate(captured_samples):
        # sample_data: (3, T, V) -> 取一帧 (3, V)
        skeleton_frame = sample_data[:, frame_idx, :]
        
        ax = fig.add_subplot(1, 6, i+1, projection='3d')
        title = f"Step {t_val}" if isinstance(t_val, int) else t_val
        plot_skeleton_3d(ax, skeleton_frame, title=title)

    file_stem = args.output_name if args.output_name else f'denoising_process{idx}'
    save_path = build_save_path(file_stem, args.save_format)
    plt.tight_layout()
    plt.savefig(save_path, format=args.save_format, dpi=300)
    print(f"✅ Saved to {save_path}")

# =====================================================================
# 任务 2: t-SNE 特征可视化
# =====================================================================
def run_tsne_viz(args):
    print("🎨 Generating t-SNE Visualization...")

    # --- NTU-60 动作名称列表 (Index 0-59) ---
    NTU60_NAMES = [
        "Drink water", "Eat meal/snack", "Brushing teeth", "Brushing hair", "Drop", "Pickup", 
        "Throw", "Sitting down", "Standing up", "Clapping", "Reading", "Writing", 
        "Tear up paper", "Wear jacket", "Take off jacket", "Wear a shoe", "Take off a shoe", 
        "Wear on glasses", "Take off glasses", "Put on a hat/cap", "Take off a hat/cap", 
        "Cheer up", "Hand waving", "Kicking something", "Reach into pocket", 
        "Hopping", "Jump up", "Phone call", "Playing with phone", "Typing", 
        "Pointing", "Taking a selfie", "Check time", "Rub two hands", "Nod head/bow", 
        "Shake head", "Wipe face", "Salute", "Put palms together", "Cross hands", 
        "Sneeze/cough", "Staggering", "Falling", "Touch head", "Touch chest", 
        "Touch back", "Touch neck", "Nausea/vomiting", "Use a fan", "Punching/slapping", 
        "Kicking other", "Pushing other", "Pat on back", "Point finger at other", 
        "Hugging", "Giving something", "Touch other's pocket", "Handshaking", 
        "Walk towards", "Walk apart"
    ]
    
    # 【重要】为了图表清晰，强烈建议只选 10-20 个类
    # 这里选择前 10 类作为示例 (你可以修改这个列表来选择你想展示的动作)
    TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"   Selecting classes IDs: {TARGET_CLASSES}")
    print(f"   Corresponding names: {[NTU60_NAMES[i] for i in TARGET_CLASSES]}")
    
    dataset = NTU_Feeder(args.data_path, split=args.split, window_size=args.max_frames, modality=args.modality)
    
    # 筛选子集
    indices = [i for i in range(len(dataset)) if dataset.label[i] in TARGET_CLASSES]
    # 为了显示不要太慢，最多取 1000 个点
    if len(indices) > 1000:
        # 随机抽样以保持分布均匀
        np.random.seed(42) # 固定随机种子保证可复现
        indices = np.random.choice(indices, 1000, replace=False)
    
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
    
    # 定义提取特征的辅助函数 (保持不变)
    def get_features(model_state):
        model = HAR_Classifier_Model(
            bone_pairs=[],
            num_classes=args.num_classes,
            num_joints=args.num_joints,
            in_channels=3,
            max_frames=args.max_frames,
            model_channels=args.model_channels,
            mamba_d_state=16,
            modality=args.modality
        ).to(DEVICE)
        
        if model_state == 'trained':
            if os.path.exists(args.phase2_ckpt):
                model.load_state_dict(torch.load(args.phase2_ckpt, map_location=DEVICE))
                print("   Loaded Trained Weights.")
            else:
                print("❌ Trained checkpoint not found! Please check path.")
                sys.exit(1)
        else:
            print("   Using Random Weights (Untrained).")
            
        model.eval()
        features = []
        labels_list = []
        
        with torch.no_grad():
            for batch, label, _ in tqdm(loader, desc=f"Extracting ({model_state})"):
                data = batch.float().to(DEVICE)
                zero_t = torch.zeros(data.size(0), device=DEVICE).long()
                t_emb = model.time_embed(zero_t)
                h_mid, _ = model.encoder(data, t_emb, model.A)
                feat = h_mid.mean(dim=(2, 3))
                features.append(feat.cpu().numpy())
                labels_list.append(label.numpy())
                
        return np.concatenate(features), np.concatenate(labels_list)

    # 1. 提取特征和数字标签
    feat_rand, label_num_rand = get_features('random')
    feat_train, label_num_train = get_features('trained')
    
    # 【核心修改】将数字标签转换为文字标签列表
    label_str_rand = [NTU60_NAMES[i] for i in label_num_rand]
    label_str_train = [NTU60_NAMES[i] for i in label_num_train]
    
    # 2. t-SNE 降维
    print("   Running t-SNE (this may take a moment)...")
    # 使用同样的随机种子，保证两张图的整体布局相似，便于对比
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    emb_rand = tsne.fit_transform(feat_rand)
    emb_train = tsne.fit_transform(feat_train)
    
    # 3. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) # 稍微调宽一点给图例留位置
    
    # 使用固定顺序的类别列表，保证图例颜色一致
    hue_order = [NTU60_NAMES[i] for i in sorted(TARGET_CLASSES)]
    palette = sns.color_palette("hsv", len(TARGET_CLASSES)) # 使用鲜艳的颜色
    
    # 图 1: Without Pre-training (使用文字标签作为 hue)
    sns.scatterplot(x=emb_rand[:,0], y=emb_rand[:,1], hue=label_str_rand, hue_order=hue_order, palette=palette, ax=axes[0], legend=False, s=40, alpha=0.7)
    axes[0].set_title("(a) Without Pre-training (Random)", fontsize=16)
    axes[0].axis('off')
    
    # 图 2: With Pre-training (使用文字标签作为 hue)
    sns.scatterplot(x=emb_train[:,0], y=emb_train[:,1], hue=label_str_train, hue_order=hue_order, palette=palette, ax=axes[1], legend='full', s=40, alpha=0.7)
    axes[1].set_title("(b) With Pre-training (Ours)", fontsize=16)
    axes[1].axis('off')
    
    # 美化图例
    # 将图例移到外侧，并设置标题
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., title="Action Name", fontsize=12, title_fontsize=14, markerscale=1.5)
    
    file_stem = args.output_name if args.output_name else 'tsne_comparison_named'
    save_path = build_save_path(file_stem, args.save_format)
    plt.tight_layout()
    # 调整布局，防止图例被截断
    plt.subplots_adjust(right=0.85) 
    plt.savefig(save_path, format=args.save_format, dpi=300, bbox_inches='tight') # bbox_inches='tight' 确保保存完整
    print(f"✅ Saved to {save_path}")

# =====================================================================
# 任务 3: 混淆矩阵可视化
# =====================================================================
def run_cm_viz(args):
    print("🎨 Generating Confusion Matrix...")
    
    # --- NTU-60 动作名称列表 (Index 0-59) ---
    NTU60_NAMES = [
        "Drink water", "Eat meal/snack", "Brushing teeth", "Brushing hair", "Drop", "Pickup", 
        "Throw", "Sitting down", "Standing up", "Clapping", "Reading", "Writing", 
        "Tear up paper", "Wear jacket", "Take off jacket", "Wear a shoe", "Take off a shoe", 
        "Wear on glasses", "Take off glasses", "Put on a hat/cap", "Take off a hat/cap", 
        "Cheer up", "Hand waving", "Kicking something", "Reach into pocket", 
        "Hopping", "Jump up", "Phone call", "Playing with phone", "Typing", 
        "Pointing", "Taking a selfie", "Check time", "Rub two hands", "Nod head/bow", 
        "Shake head", "Wipe face", "Salute", "Put palms together", "Cross hands", 
        "Sneeze/cough", "Staggering", "Falling", "Touch head", "Touch chest", 
        "Touch back", "Touch neck", "Nausea/vomiting", "Use a fan", "Punching/slapping", 
        "Kicking other", "Pushing other", "Pat on back", "Point finger at other", 
        "Hugging", "Giving something", "Touch other's pocket", "Handshaking", 
        "Walk towards", "Walk apart"
    ]

    dataset = NTU_Feeder(args.data_path, split=args.split, window_size=args.max_frames, modality=args.modality)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
    
    model = HAR_Classifier_Model(
        bone_pairs=[],
        num_classes=args.num_classes,
        num_joints=args.num_joints,
        in_channels=3,
        max_frames=args.max_frames,
        model_channels=args.model_channels,
        mamba_d_state=16,
        modality=args.modality
    ).to(DEVICE)
    
    if os.path.exists(args.phase2_ckpt):
        model.load_state_dict(torch.load(args.phase2_ckpt, map_location=DEVICE))
        print(f"   Loaded weights: {args.phase2_ckpt}")
    else:
        print(f"❌ Checkpoint not found: {args.phase2_ckpt}")
        return
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("   Running Inference on Full Test Set...")
    with torch.no_grad():
        for batch, label, _ in tqdm(loader):
            data = batch.float().to(DEVICE)
            logits = model(data)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 归一化 (Row-wise normalization) -> 变成百分比
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # -------------------------------------------------------
    # 图 1: 全量混淆矩阵 (60x60) - 尽量让标签看清
    # -------------------------------------------------------
    print("   Plotting Full Matrix (60x60)...")
    plt.figure(figsize=(24, 20)) # 【关键】画布设得非常大
    
    sns.heatmap(cm_norm, cmap='Blues', 
                xticklabels=NTU60_NAMES, 
                yticklabels=NTU60_NAMES,
                cbar_kws={'shrink': 0.5}) # 颜色条短一点
    
    plt.xticks(fontsize=8, rotation=90) # 字体调小，旋转90度
    plt.yticks(fontsize=8, rotation=0)
    
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title(f'Confusion Matrix - Full ({args.modality})', fontsize=20)
    
    file_stem_full = args.output_name if args.output_name else 'confusion_matrix_full'
    save_path = build_save_path(file_stem_full, args.save_format)
    plt.tight_layout()
    plt.savefig(save_path, format=args.save_format, dpi=300)
    print(f"✅ Saved Full Matrix to {save_path}")
    
    # -------------------------------------------------------
    # 图 2: 局部高难度样本放大 (Zoom-in) - 论文分析专用
    # -------------------------------------------------------
    print("   Plotting Zoom-in Matrix (Hard Classes)...")
    
    # 定义我们要重点观察的“易混淆”对
    # Reading(10)/Writing(11), Wear Shoe(15)/Take off Shoe(16), 
    # Wear Glasses(17)/Take off Glasses(18), Rub Hands(33)/Clapping(9)
    target_indices = [9, 10, 11, 15, 16, 17, 18, 33]
    
    # 根据索引提取名称和子矩阵
    subset_names = [NTU60_NAMES[i] for i in target_indices]
    
    # 提取子矩阵 (只取这些行和列)
    # cm_norm[行索引][:, 列索引]
    cm_sub = cm_norm[np.ix_(target_indices, target_indices)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_sub, annot=True, fmt=".2f", cmap='Reds', # 换个红色系，醒目
                xticklabels=subset_names, 
                yticklabels=subset_names,
                linewidths=1, linecolor='gray') # 加网格线
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title('Confusion Matrix (Selected Hard Classes)', fontsize=14)
    
    file_stem_sub = (
        f"{args.output_name}_hard_classes" if args.output_name else 'confusion_matrix_hard_classes'
    )
    save_path_sub = build_save_path(file_stem_sub, args.save_format)
    plt.tight_layout()
    plt.savefig(save_path_sub, format=args.save_format, dpi=300)
    print(f"✅ Saved Zoom-in Matrix to {save_path_sub}")

# =====================================================================
# Main Interface
# =====================================================================
if __name__ == '__main__':
    args = get_args()
    
    if args.task == 'denoise':
        run_denoising_viz(args)
    elif args.task == 'tsne':
        run_tsne_viz(args)
    elif args.task == 'cm':
        run_cm_viz(args)
    else:
        print("Invalid task.")
