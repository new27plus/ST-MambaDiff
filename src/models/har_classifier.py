import torch
import torch.nn as nn
import torch.nn.functional as F

from .st_unet_denoiser import ST_Unet_Encoder, TimestepEmbedding, LegacyGraph

try:
    from graph.ntu_rgb_d import Graph as NTUGraph
except ImportError:
    NTUGraph = None

# ----------------------------------------------------------------------
# 1. 增强版分类头
# ----------------------------------------------------------------------
class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=512, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU() 
        self.dropout = nn.Dropout(p=dropout) 
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=3)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x_cat = torch.cat([x_max, x_avg], dim=1)
        x = self.fc1(x_cat)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# ----------------------------------------------------------------------
# 2. 最终 HAR 分类模型 (接收 modality 参数)
# ----------------------------------------------------------------------
class HAR_Classifier_Model(nn.Module):
    def __init__(self,
                 bone_pairs,
                 num_classes,
                 num_joints=20,
                 in_channels=3,
                 max_frames=64,
                 model_channels=128,
                 mamba_d_state=16,
                 dropout=0.5,
                 modality='joint',  # 【新增】接收模态参数
                 temporal_operator='auto'  # 【消融】auto/mamba/dwconv/transformer
                 ):
        super(HAR_Classifier_Model, self).__init__()
        
        if num_joints == 25 and NTUGraph is not None:
            print(f">>> [Model] Loaded SOTA Spatial Graph for NTU ({num_joints} joints)")
            graph = NTUGraph(labeling_mode='spatial')
            A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        else:
            print(f">>> [Model] Loaded Legacy Graph for UCLA/Other ({num_joints} joints)")
            graph = LegacyGraph(feeder_bone_pairs=bone_pairs, num_joints=num_joints)
            A = graph.A 
            if not isinstance(A, torch.Tensor):
                A = torch.tensor(A, dtype=torch.float32, requires_grad=False)

        self.register_buffer('A', A)
        
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        self.encoder = ST_Unet_Encoder(
            in_channels=in_channels,
            model_channels=model_channels,
            time_embed_dim=self.time_embed_dim,
            A=self.A,
            mamba_d_state=mamba_d_state,
            modality=modality, # 【关键】传递模态给 Encoder
            temporal_operator=temporal_operator
        )
        
        encoder_out_channels = model_channels * 4 
        self.classification_head = ClassificationHead(
            in_channels=encoder_out_channels, 
            num_classes=num_classes,
            hidden_dim=encoder_out_channels,
            dropout=dropout
        )

    def forward(self, x):
        t_emb = None 
        h_mid, _ = self.encoder(x, t_emb, self.A)
        logits = self.classification_head(h_mid)
        return logits