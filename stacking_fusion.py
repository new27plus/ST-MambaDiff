import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# 引入 sklearn 库
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# 确保能导入本地模型
sys.path.append(os.getcwd())
from models.har_classifier import HAR_Classifier_Model
from feeders.ucla_feeder import Feeder 

# ==========================================
# ⚙️ 配置 (Configuration)
# ==========================================
DATA_PATH = 'datasets/NW-UCLA/all_sqe'
VAL_LABEL_PATH = 'datasets/NW-UCLA/ucla_val_data.json'
CHECKPOINT_PATH = './experiments/UCLA/checkpoints'

BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ⚠️ 必须与你 SOTA 训练时的配置完全一致！
MODEL_ARGS = {
    'num_classes': 10, 
    'num_joints': 20, 
    'in_channels': 3,
    'max_frames': 52, 
    'model_channels': 128,  # SOTA 配置
    'mamba_d_state': 16,    # SOTA 配置
}

BONE_PAIRS = [(0,0)] 

# ==========================================
# 📥 数据加载函数
# ==========================================
def get_logits_and_labels(modality):
    print(f"Extracting features for {modality}...")
    
    # 初始化模型
    model = HAR_Classifier_Model(bone_pairs=BONE_PAIRS, **MODEL_ARGS).to(DEVICE)
    
    # 加载权重
    ckpt_path = os.path.join(CHECKPOINT_PATH, f'phase2_{modality}_best_classifier.pt')
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found for {modality} at {ckpt_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    # 加载数据
    dataset = Feeder(data_path=DATA_PATH, label_path=VAL_LABEL_PATH, modality=modality, repeat=1)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for data, label, _ in tqdm(loader, desc=f"Inferencing {modality}", leave=False):
            data = data.squeeze(-1).to(DEVICE)
            
            # 模型前向传播 (t_emb=None 已在模型内部处理)
            out = model(data)
            
            logits_list.append(out.cpu().numpy())
            labels_list.append(label.numpy())
            
    return np.concatenate(logits_list), np.concatenate(labels_list)

# ==========================================
# 🚀 主程序 (Stacking Logic)
# ==========================================
def main():
    modes = ['joint', 'bone', 'motion', 'bone-motion']
    
    # 1. 准备 Stacking 的输入数据
    # 我们需要构建一个矩阵 X，形状为 [N_samples, 40]
    # 40 = 4个模态 * 10个类别概率
    X_parts = []
    y = None
    
    print("\n>>> [Step 1] Gathering Logits from all streams...")
    
    for m in modes:
        logits, labels = get_logits_and_labels(m)
        X_parts.append(logits)
        if y is None: y = labels
        
        # 打印单流准确率供参考
        acc = (np.argmax(logits, axis=1) == labels).sum() / len(labels)
        print(f"    -> {m.ljust(12)} Single Acc: {acc:.4f}")
        
    # 拼接特征: [N, 10] + [N, 10] + ... -> [N, 40]
    X = np.concatenate(X_parts, axis=1)
    
    print("\n>>> [Step 2] Starting Stacking Fusion (Meta-Learning)...")
    print("    Strategy: Logistic Regression with 5-Fold Cross-Validation")
    
    # 2. 交叉验证评估 (这是最真实的 SOTA 成绩)
    # 我们使用 StratifiedKFold 保证每一折的类别分布均匀
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    
    # 正则化强度 C: 小一点(0.1)表示更强的正则化，防止在这个小数据集上过拟合
    meta_model_config = {'max_iter': 2000, 'C': 0.1, 'solver': 'lbfgs'}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(**meta_model_config)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv_accs.append(acc)
        # print(f"    Fold {fold+1} Acc: {acc:.4f}")
        
    mean_acc = np.mean(cv_accs)
    std_acc = np.std(cv_accs)
    
    print("\n" + "="*50)
    print(f"🏆 STACKING CV RESULT (True SOTA): {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"   Accuracy Percentage: {mean_acc*100:.2f}%")
    print("="*50)

    # 3. 全量拟合分析 (仅用于分析贡献度，分数会虚高)
    final_clf = LogisticRegression(**meta_model_config)
    final_clf.fit(X, y)
    
    # 分析权重贡献 (Coefficient Analysis)
    # coef_ shape: [10_classes, 40_features]
    # 我们取绝对值平均，看看哪个模态的特征对决策最重要
    coefs = np.abs(final_clf.coef_) 
    # Reshape to [10_classes, 4_modes, 10_features_per_mode]
    coefs_reshaped = coefs.reshape(10, 4, 10)
    # Sum over classes and features -> [4_modes]
    importance = np.sum(coefs_reshaped, axis=(0, 2))
    
    # Normalize to 100%
    importance = importance / np.sum(importance) * 100
    
    print("\n📊 Estimated Stream Contribution (Stacking Weights):")
    for i, m in enumerate(modes):
        print(f"  {m.ljust(12)}: {importance[i]:.2f}%")
        
    if mean_acc > 0.9612:
        print("\n🚀 NEW RECORD! Stacking outperformed linear weighting.")
    else:
        print("\nℹ️ Result Analysis: Stacking performed similarly to linear fusion.")
        print("   This suggests the individual models are highly correlated.")

if __name__ == "__main__":
    main()