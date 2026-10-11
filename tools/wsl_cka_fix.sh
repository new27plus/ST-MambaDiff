#!/usr/bin/env bash
# 修正版 CKA 一次性诊断（分子带平方的正确公式），对比三对表征
cd /mnt/c/Users/14811/Documents/Personal/Study/School/论文自己/第三篇/Experment
~/anaconda3/envs/mamba/bin/python - <<'EOF'
import torch, glob, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, './src')
from src.models.har_classifier import HAR_Classifier_Model

BONE=[(1,0),(2,1),(3,2),(4,2),(5,4),(6,5),(7,6),(8,7),(9,8),(10,9),(11,10),(12,0),
      (13,12),(14,13),(15,14),(16,15),(16,0),(17,16),(18,17),(19,18)]

def cka(X, Y):
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, 'fro') ** 2          # ← 分子带平方（修正点）
    return xy / (np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro') + 1e-12)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build():
    return HAR_Classifier_Model(bone_pairs=BONE, num_classes=10, num_joints=20,
        in_channels=3, max_frames=52, model_channels=128, mamba_d_state=16,
        dropout=0.0, modality='joint').to(dev).eval()

from feeders.ucla_feeder import Feeder
from torch.utils.data import DataLoader
te = Feeder(data_path='../Code/datasets/NW-UCLA/all_sqe',
            label_path='../Code/datasets/NW-UCLA/ucla_val_data.json', modality='joint', repeat=1)
loader = DataLoader(te, batch_size=64, shuffle=False, num_workers=2)

def feats(m):
    fs = []
    with torch.no_grad():
        for b, _, _ in loader:
            x = b.float().squeeze(-1).to(dev)
            t_emb = m.time_embed(torch.zeros(x.size(0), device=dev).long())
            h, _ = m.encoder(x, t_emb, m.A)
            fs.append(h.mean(dim=(2,3)).flatten(1).cpu().numpy())
            if sum(f.shape[0] for f in fs) >= 464: break
    return np.concatenate(fs)

p1 = sorted(glob.glob('runs/ucla_final/joint/seed0/*/checkpoints/phase1_joint_best_acc.pt'))[-1]
p2_full  = sorted(glob.glob('runs/ucla_final/joint/seed0/*/checkpoints/phase2_joint_best_classifier.pt'))[-1]
p2_gentle= sorted(glob.glob('runs/ucla_final_gentle/joint/seed0/*/checkpoints/phase2_joint_best_classifier.pt'))[-1]

m1 = build(); m1.load_state_dict(torch.load(p1, map_location=dev, weights_only=True), strict=False)
F1 = feats(m1)
F_rand = feats(build())   # 随机初始化地板
for tag, pat in [('全参数微调(E1 phase2)', p2_full), ('温和微调(E4b phase2)', p2_gentle)]:
    m2 = build(); m2.load_state_dict(torch.load(pat, map_location=dev, weights_only=True), strict=True)
    print(f'修正CKA( phase1 vs {tag} ) = {cka(F1, feats(m2)):.4f}')
print(f'修正CKA( phase1 vs 随机初始化 ) = {cka(F1, F_rand):.4f}   ← 地板参照')
EOF
