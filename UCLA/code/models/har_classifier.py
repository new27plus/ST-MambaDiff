import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 导入核心组件 ---
from .st_unet_denoiser import ST_Unet_Encoder, Graph, TimestepEmbedding

# ----------------------------------------------------------------------
# 1. 增强版分类头 (SOTA Head: Double Pooling + MLP)
# ----------------------------------------------------------------------
class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=512):
        super(ClassificationHead, self).__init__()
        
        # 1. MLP 结构
        # 输入维度是 in_channels * 2 (因为 Max + Avg 拼接)
        self.fc1 = nn.Linear(in_channels * 2, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU() 
        self.dropout = nn.Dropout(p=0.5) # 强正则化
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input: (B, C, T, V)
        
        # 1. 空间聚合: (B, C, T, V) -> (B, C, T)
        x = x.mean(dim=3)
        
        # 2. 时间聚合: Max + Avg (关键！捕捉峰值和平均)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1) # (B, C)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # (B, C)
        
        # 拼接
        x_cat = torch.cat([x_max, x_avg], dim=1)
        
        # 3. MLP 分类
        x = self.fc1(x_cat)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

# ----------------------------------------------------------------------
# 2. 最终 HAR 分类模型 (Neutral Time Condition)
# ----------------------------------------------------------------------
class HAR_Classifier_Model(nn.Module):
    def __init__(self, 
                 bone_pairs, 
                 num_classes, 
                 num_joints=20, 
                 in_channels=3, 
                 max_frames=52, 
                 model_channels=128,  # SOTA 配置
                 mamba_d_state=16     # SOTA 配置
                 ):
        super(HAR_Classifier_Model, self).__init__()
        
        # 1. 图
        self.graph = Graph(feeder_bone_pairs=bone_pairs, num_joints=num_joints)
        self.register_buffer('A', self.graph.A) 
        
        # 2. 时间嵌入 (虽然这里不推理 t，但为了加载权重不报错，还是定义着比较安全)
        #    或者我们可以保留它但不使用，只为了 load_state_dict 兼容性
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # 3. Encoder
        self.encoder = ST_Unet_Encoder(
            in_channels=in_channels, 
            model_channels=model_channels, 
            time_embed_dim=self.time_embed_dim, 
            A=self.A, 
            mamba_d_state=mamba_d_state
        )
        
        # 4. 分类头
        encoder_out_channels = model_channels * 4 
        self.classification_head = ClassificationHead(
            in_channels=encoder_out_channels, 
            num_classes=num_classes,
            hidden_dim=encoder_out_channels 
        )

    def forward(self, x):
        """
        x: (B, C, T, V)
        """
        
        # 【核心修改】直接传 None！
        # 这告诉 Encoder："不要加入任何时间/噪声偏置，只处理数据本身。"
        # 这比 t=0 更中性，更能利用 Encoder 的通用结构特征。
        t_emb = None 
        
        # 2. Encoder 特征提取
        # h_mid shape: (B, C_out, T_mid, V)
        h_mid, _ = self.encoder(x, t_emb, self.A)
        
        # 3. 分类
        logits = self.classification_head(h_mid)
        
        return logits