import torch, glob, sys
sys.path.insert(0, '.')
from src.models.har_classifier import HAR_Classifier_Model
BONE=[(1,0),(2,1),(3,2),(4,2),(5,4),(6,5),(7,6),(8,7),(9,8),(10,9),(11,10),(12,0),(13,12),(14,13),(15,14),(16,15),(16,0),(17,16),(18,17),(19,18)]
for m in ['joint','bone']:
    ck = sorted(glob.glob(f'runs/ucla_final/{m}/seed0/*/checkpoints/phase1_{m}_best_acc.pt'))[-1]
    sd = torch.load(ck, map_location='cpu', weights_only=True)
    model = HAR_Classifier_Model(bone_pairs=BONE, num_classes=10, num_joints=20,
        in_channels=3, max_frames=52, model_channels=128, mamba_d_state=16, modality=m)
    md = model.state_dict()
    want = {k for k in sd if k.startswith(('encoder.','time_embed.'))}
    hit  = {k for k in want if k in md and md[k].shape == sd[k].shape}
    print(f'[{m}] phase1键={len(sd)} encoder+time_embed键={len(want)} 实际加载={len(hit)} 缺失={len(want)-len(hit)}')
