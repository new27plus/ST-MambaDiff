import torch
import torch.nn as nn
import numpy as np
from mamba_ssm import Mamba

# 导入 TimestepEmbedding
from .embeddings import TimestepEmbedding

# ----------------------------------------------------------------------
# 0. 基础组件: RMSNorm
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return x_normed * self.weight

# ----------------------------------------------------------------------
# 1. 图结构 (不变)
# ----------------------------------------------------------------------
class Graph:
    def __init__(self, feeder_bone_pairs, num_joints=20, strategy='spatial'):
        self.num_joints = num_joints
        self.bone_pairs = feeder_bone_pairs
        self.strategy = strategy
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((3, self.num_joints, self.num_joints), dtype=np.float32)
        A[0] = np.eye(self.num_joints, dtype=np.float32)
        for child, parent in self.bone_pairs:
            A[1, child, parent] = 1
        for child, parent in self.bone_pairs:
            A[2, parent, child] = 1
        
        for i in range(3):
            D = np.sum(A[i], axis=1)
            D[D == 0] = 1
            D_inv = np.diag(D**-0.5)
            A[i] = D_inv @ A[i] @ D_inv
        return torch.from_numpy(A)

# ----------------------------------------------------------------------
# 2. 空间处理: Advanced Shift-GCN (保留这个优秀的归纳偏置)
# ----------------------------------------------------------------------
class AdvancedShift_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, max_neighbors=2):
        super(AdvancedShift_GCN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_partitions = A.shape[0]
        self.num_joints = A.shape[1]
        
        c_per_part = in_channels // self.num_partitions
        c_per_neighbor = c_per_part // max_neighbors
        
        shift_index = torch.arange(self.num_joints).unsqueeze(0).repeat(in_channels, 1)
        A_binary = (A > 0.001).float()
        
        for p in range(self.num_partitions):
            part_start = p * c_per_part
            part_end = (p + 1) * c_per_part if p < self.num_partitions - 1 else in_channels
            if p == 0: continue
            for v in range(self.num_joints):
                neighbors = A_binary[p, v].nonzero(as_tuple=False).flatten()
                if len(neighbors) == 0:
                    current_neighbors = [v] * max_neighbors
                else:
                    current_neighbors = neighbors.tolist()
                    while len(current_neighbors) < max_neighbors:
                        current_neighbors.append(current_neighbors[0])
                    current_neighbors = current_neighbors[:max_neighbors]
                for k, neighbor_idx in enumerate(current_neighbors):
                    sub_start = part_start + k * c_per_neighbor
                    if k == max_neighbors - 1: sub_end = part_end 
                    else: sub_end = part_start + (k + 1) * c_per_neighbor
                    if sub_start < sub_end:
                        shift_index[sub_start:sub_end, v] = neighbor_idx

        self.register_buffer('shift_index', shift_index.long())
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        if not residual: self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1: self.residual = lambda x: x
        else: self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))

    def forward(self, x, A=None):
        res = self.residual(x)
        B, C, T, V = x.shape
        index = self.shift_index.view(1, C, 1, V).expand(B, C, T, V)
        x_shifted = torch.gather(x, dim=3, index=index)
        x_out = self.conv(x_shifted)
        return x_out + res

# ----------------------------------------------------------------------
# 3. 时间处理: Lightweight Per-Joint Mamba (防止过拟合)
# ----------------------------------------------------------------------
class TemporalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=1): # 【修改】d_state=8, expand=1
        super(TemporalMambaBlock, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        B, C, T, V = x.shape
        x_permuted = x.permute(0, 3, 2, 1).contiguous()
        x_reshaped = x_permuted.view(B * V, T, C)
        x_normed = self.norm(x_reshaped)
        mamba_out = self.mamba(x_normed) 
        mamba_out_reshaped = mamba_out.view(B, V, T, C)
        x_final = mamba_out_reshaped.permute(0, 3, 2, 1).contiguous()
        return x_final

# ----------------------------------------------------------------------
# 4. 下采样 (ResDownsample - 保留)
# ----------------------------------------------------------------------
class ResTemporalDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,1), stride=(2,1), padding=(1,0)):
        super(ResTemporalDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.AvgPool2d(kernel_size=(2,1), stride=(2,1))
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.proj(self.pool(x))

# ----------------------------------------------------------------------
# 5. 核心 U-Net 构建块 (GroupNorm 版)
# ----------------------------------------------------------------------
class ST_Unet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, A, mamba_d_state=8):
        super(ST_Unet_Block, self).__init__()
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        if in_channels == out_channels: self.conv1 = nn.Identity()
        else: self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 1. 空间
        self.spatial_gcn = AdvancedShift_GCN(out_channels, out_channels, A)
        
        # 2. 时间 (Lightweight Mamba)
        self.temporal_mamba = TemporalMambaBlock(d_model=out_channels, d_state=mamba_d_state, expand=1)
        
        # 【修改】使用 GroupNorm 替代 BatchNorm2d
        # GroupNorm 在 Diffusion 任务中更稳定，且不受 BatchSize 影响
        # num_groups=8 是一个经验上比较稳妥的值 (适配 64, 128, 256 通道)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.2) 

    def forward(self, x, t_emb, A): 
        x = self.conv1(x) 
        
        # 仅当 t_emb 存在时注入 (Encoder 注入, Decoder 不注入)
        if t_emb is not None:
            t_emb_proj = self.time_proj(t_emb)[:, :, None, None]
            x = x + t_emb_proj
        
        x_s = self.spatial_gcn(x, A)
        x_st = self.temporal_mamba(x_s)
        
        # Norm -> Act -> Dropout
        x_out = self.dropout(self.act(self.norm(x_st + x)))
        return x_out

# ----------------------------------------------------------------------
# 6. Encoder (结构不变，参数调整)
# ----------------------------------------------------------------------
class ST_Unet_Encoder(nn.Module):
    def __init__(self, in_channels, model_channels, time_embed_dim, A, mamba_d_state):
        super(ST_Unet_Encoder, self).__init__()
        C1, C2, C3 = model_channels, model_channels * 2, model_channels * 4
        
        self.enc_in = ST_Unet_Block(in_channels, C1, time_embed_dim, A, mamba_d_state)
        self.enc_block1 = ST_Unet_Block(C1, C1, time_embed_dim, A, mamba_d_state)
        self.down1 = ResTemporalDownsample(C1, C1)

        self.enc_block2 = ST_Unet_Block(C1, C2, time_embed_dim, A, mamba_d_state)
        self.enc_block3 = ST_Unet_Block(C2, C2, time_embed_dim, A, mamba_d_state)
        self.down2 = ResTemporalDownsample(C2, C2)

        self.mid_block1 = ST_Unet_Block(C2, C3, time_embed_dim, A, mamba_d_state)
        self.mid_block2 = ST_Unet_Block(C3, C3, time_embed_dim, A, mamba_d_state)

    def forward(self, x, t_emb, A):
        skips = []
        h1 = self.enc_in(x, t_emb, A)
        h1 = self.enc_block1(h1, t_emb, A) 
        skips.append(h1)
        
        h_down1 = self.down1(h1) 
        h2 = self.enc_block2(h_down1, t_emb, A)
        h2 = self.enc_block3(h2, t_emb, A) 
        skips.append(h2)

        h_down2 = self.down2(h2)
        h_mid = self.mid_block1(h_down2, t_emb, A)
        h_mid = self.mid_block2(h_mid, t_emb, A) 
        
        return h_mid, skips

# ----------------------------------------------------------------------
# 7. Decoder (支持 t_emb=None)
# ----------------------------------------------------------------------
class ST_Unet_Decoder(nn.Module):
    def __init__(self, out_channels, model_channels, time_embed_dim, A, mamba_d_state):
        super(ST_Unet_Decoder, self).__init__()
        C1, C2, C3 = model_channels, model_channels * 2, model_channels * 4

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(2, 1), mode='nearest'), nn.Conv2d(C3, C2, (3,1), padding=(1,0)))
        self.dec_block1 = ST_Unet_Block(C2 * 2, C2, time_embed_dim, A, mamba_d_state)
        self.dec_block2 = ST_Unet_Block(C2, C2, time_embed_dim, A, mamba_d_state)
        
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(2, 1), mode='nearest'), nn.Conv2d(C2, C1, (3,1), padding=(1,0)))
        self.dec_block3 = ST_Unet_Block(C1 * 2, C1, time_embed_dim, A, mamba_d_state)
        self.dec_block4 = ST_Unet_Block(C1, C1, time_embed_dim, A, mamba_d_state)
        
        self.out_head = nn.Conv2d(C1, out_channels, kernel_size=1)

    def forward(self, h_mid, skip_connections, t_emb, A):
        h1 = skip_connections.pop() 
        h2 = skip_connections.pop() 
        
        d1 = self.up1(h_mid)
        d1 = torch.cat([d1, h1], dim=1) 
        d1 = self.dec_block1(d1, t_emb, A) # t_emb 可能为 None
        d1 = self.dec_block2(d1, t_emb, A) 
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, h2], dim=1) 
        d2 = self.dec_block3(d2, t_emb, A)
        d2 = self.dec_block4(d2, t_emb, A) 
        
        out = self.out_head(d2)
        return out

# ----------------------------------------------------------------------
# 8. 最终模型 Wrapper (逻辑控制核心)
# ----------------------------------------------------------------------
class ST_Unet_Denoising_Model(nn.Module):
    def __init__(self, 
                 bone_pairs, 
                 num_joints=20, 
                 in_channels=3, 
                 max_frames=52, 
                 model_channels=64, # 【建议】改回 64，对于 NW-UCLA 足够了，且更聚焦语义
                 mamba_d_state=8    # 【建议】改小到 8
                 ):
        super(ST_Unet_Denoising_Model, self).__init__()
        
        self.graph = Graph(feeder_bone_pairs=bone_pairs, num_joints=num_joints)
        self.register_buffer('A', self.graph.A) 
        
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        self.encoder = ST_Unet_Encoder(in_channels, model_channels, self.time_embed_dim, self.A, mamba_d_state)
        self.decoder = ST_Unet_Decoder(in_channels, model_channels, self.time_embed_dim, self.A, mamba_d_state)

    def forward(self, x, t):
        # 1. 正常计算 Time Embedding
        t_emb = self.time_embed(t)
        
        # 2. Encoder 正常注入 t，获取语义特征 h_mid
        h_mid, skips = self.encoder(x, t_emb, self.A)
        
        # 3. 【核心策略】Decoder 不给 t！
        # 这强迫 h_mid 必须包含足够的信息来还原动作，而不能依赖 t 来“作弊”
        out = self.decoder(h_mid, skips, None, self.A)
        
        return out