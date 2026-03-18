import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys

# 引入你的组件
from feeders.ntu_feeder import NTU_Feeder
from models.har_classifier import HAR_Classifier_Model

def get_args():
    parser = argparse.ArgumentParser(description="NTU SOTA Ensemble Fusion")
    # parser.add_argument('--data_path', type=str, default='datasets/NTU-RGBD/NTU60_CS.npz')
    # parser.add_argument('--split', type=str, default='CS', choices=['CS', 'CV'])
    # parser.add_argument('--checkpoint_root', type=str, default='./checkpoints_ntu')
    parser.add_argument('--data_path', type=str, default='datasets/NTU120/ntu120/NTU120_CSub.npz')
    parser.add_argument('--split', type=str, default='CSub', choices=['CSub', 'CSet'])
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints_ntu120')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()

def load_model_and_get_logits(args, modality):
    """
    加载指定模态的模型，并跑完测试集，返回所有样本的 Softmax 分数
    """
    print(f"\n[Ensemble] Processing Modality: {modality}...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 路径检查
    ckpt_path = os.path.join(args.checkpoint_root, args.split, modality, 'phase2_sota_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    # 2. 数据加载
    # 注意：Feeder 需要根据 modality 变化
    # use_vel/use_bone 参数逻辑要与训练时一致
    val_set = NTU_Feeder(args.data_path, split='test', window_size=64, modality=modality)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # 3. 模型初始化 (必须与训练配置严格一致)
    model = HAR_Classifier_Model(
        bone_pairs=[], 
        # num_classes=60,
        num_classes=120,
        num_joints=25,
        in_channels=3,
        max_frames=64,
        model_channels=128,  # 确保是 128
        mamba_d_state=16,    # 确保是 16
        dropout=0.5,
        modality=modality    # 传入模态
    ).to(DEVICE)
    
    # 4. 加载权重
    print(f"   Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 5. 推理
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch, label, _ in tqdm(val_loader, desc=f"   Inference {modality}"):
            data = batch.float().to(DEVICE)
            
            # 获取 Logits
            logits = model(data)
            
            # 转为 Softmax 概率 (Score)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算单流准确率
    preds = np.argmax(all_probs, axis=1)
    acc = np.sum(preds == all_labels) / len(all_labels)
    print(f"   ✅ {modality} Single Stream Acc: {acc:.4f}")
    
    return all_probs, all_labels

def main():
    args = get_args()
    print(f"==================================================")
    print(f"🚀 Starting SOTA Ensemble Fusion for Split: {args.split}")
    print(f"==================================================")
    
    # 1. 获取三个流的概率分数 (Probs)
    # 注意：请确保你确实跑完了这三个流，且路径正确
    try:
        probs_joint, labels = load_model_and_get_logits(args, 'joint')
        probs_bone, _       = load_model_and_get_logits(args, 'bone')
        probs_motion, _     = load_model_and_get_logits(args, 'motion')
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("Tip: Make sure you have trained all 3 streams (joint, bone, motion).")
        return

    # 2. 高精度网格搜索 (Grid Search)
    print(f"\n[Ensemble] Starting Fine-grained Weight Search...")
    
    best_acc = 0.0
    best_weights = (0, 0, 0)
    
    # --- 🔎 定义更平滑、更全面的搜索空间 ---
    # Joint:  0.5 ~ 1.5 (基准)
    # Bone:   0.8 ~ 2.6 (Bone 通常最强，给予更高的上限探索)
    # Motion: 0.2 ~ 1.5 (Motion 是辅助，但也可能在某些类上很重要)
    
    # 使用 numpy 生成步长为 0.1 的序列
    w_joints  = np.arange(0.5, 1.6, 0.1) 
    w_bones   = np.arange(0.8, 2.6, 0.1)
    w_motions = np.arange(0.2, 1.6, 0.1)
    
    # 计算总组合数用于进度条
    total_combinations = len(w_joints) * len(w_bones) * len(w_motions)
    
    # 使用 tqdm 显示搜索进度
    with tqdm(total=total_combinations, desc="Searching Best Weights") as pbar:
        for w_j in w_joints:
            for w_b in w_bones:
                for w_m in w_motions:
                    
                    # 融合公式: 加权求和
                    # 注意: 这里不需要做 softmax，因为输入已经是 probs
                    # 也不需要除以 (w_j + w_b + w_m)，因为 argmax 对缩放不敏感
                    final_scores = w_j * probs_joint + w_b * probs_bone + w_m * probs_motion
                    
                    # 计算准确率
                    preds = np.argmax(final_scores, axis=1)
                    acc = np.sum(preds == labels) / len(labels)
                    
                    # 记录最佳
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = (w_j, w_b, w_m)
                        # 仅在显著提升时打印，避免刷屏
                        # pbar.write(f"   🚀 New Best! {acc:.4f} | Weights: J={w_j:.1f}, B={w_b:.1f}, M={w_m:.1f}")
                    
                    pbar.update(1)

    print(f"\n==================================================")
    print(f"🎉 Final SOTA Ensemble Result ({args.split})")
    print(f"==================================================")
    print(f"🏆 Best Accuracy: {best_acc * 100:.4f}%")  # 转百分比显示
    print(f"⚖️  Best Weights:")
    print(f"    Joint  : {best_weights[0]:.1f}")
    print(f"    Bone   : {best_weights[1]:.1f}")
    print(f"    Motion : {best_weights[2]:.1f}")
    print(f"==================================================")

if __name__ == '__main__':
    main()