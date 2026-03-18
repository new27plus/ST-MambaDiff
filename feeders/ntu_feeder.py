import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# =====================================================
# Utility Functions (保持不变，逻辑是正确的)
# =====================================================

def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # data_numpy: (C,T,V,M)
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), 64),
            valid_size
        )
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]

    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(
        data,
        size=(C * V * M, window),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).numpy()
    return data


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros_like(data_numpy)
    valid_frame = (data_numpy != 0).sum(axis=(0, 2, 3)) > 0
    if valid_frame.sum() == 0:
        return data_numpy

    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    size = end - begin

    if size < T:
        bias = random.randint(0, T - size)
        data_shift[:, bias:bias + size] = data_numpy[:, begin:end]
    else:
        data_shift = data_numpy

    return data_shift


def random_rot(data_numpy, theta=0.3): # 建议减小 theta，0.6 有点太大了
    data = torch.from_numpy(data_numpy)
    C, T, V, M = data.shape
    data = data.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)

    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot] * T)
    cos_r, sin_r = rot.cos(), rot.sin()

    zeros = torch.zeros(T, 1)
    ones = torch.ones(T, 1)

    rx = torch.cat([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_r[:, 0:1], sin_r[:, 0:1]], dim=-1),
        torch.stack([zeros, -sin_r[:, 0:1], cos_r[:, 0:1]], dim=-1)
    ], dim=1)

    ry = torch.cat([
        torch.stack([cos_r[:, 1:2], zeros, -sin_r[:, 1:2]], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([sin_r[:, 1:2], zeros, cos_r[:, 1:2]], dim=-1)
    ], dim=1)

    rz = torch.cat([
        torch.stack([cos_r[:, 2:3], sin_r[:, 2:3], zeros], dim=-1),
        torch.stack([-sin_r[:, 2:3], cos_r[:, 2:3], zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=1)

    rot_mat = rz @ ry @ rx
    data = torch.matmul(rot_mat, data)
    data = data.view(T, C, V, M).permute(1, 0, 2, 3).numpy()
    return data


# =====================================================
# NTU Bone Pairs (1-based) - 【关键修正】
# =====================================================
# 移除了 (21, 21)，防止 Bone 模态出现全0向量或轨迹噪声
ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (23, 8), # 注意这里删除了 (21,21)
    (24, 25), (25, 12)
)


# =====================================================
# NTU Feeder (Updated)
# =====================================================

class NTU_Feeder(Dataset):
    def __init__(
        self,
        data_path,
        split='train',
        window_size=64, # 建议设为 64
        p_interval=[0.95],
        modality='joint', # 【修改】使用字符串控制
        random_rot_flag=True
    ):
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.p_interval = p_interval
        self.modality = modality # 'joint', 'bone', 'motion'
        self.random_rot_flag = random_rot_flag
        self.load_data()

    def load_data(self):
        npz = np.load(self.data_path, mmap_mode='r')
        if self.split == 'train':
            self.data = npz['x_train']
            self.label = np.where(npz['y_train'] > 0)[1]
        else:
            self.data = npz['x_test']
            self.label = np.where(npz['y_test'] > 0)[1]

        N, T, _ = self.data.shape
        # reshape 到 (N, C, T, V, M)
        self.data = self.data.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = np.array(self.data[index])  # (C,T,V,M)
        label = self.label[index]

        # ===============================
        # 1. 有效帧计算
        # ===============================
        valid_frame = np.sum(data.sum(axis=0).sum(axis=-1).sum(axis=-1) != 0)

        # ===============================
        # 2. Crop & Resize
        # ===============================
        p = self.p_interval if self.split == 'train' else [1]
        data = valid_crop_resize(data, valid_frame, p, self.window_size)

        # ===============================
        # 3. 几何数据增强 (仅作用于 Joint)
        # ===============================
        # 这里的顺序非常重要：先对 Joint 做旋转/位移，再基于变换后的 Joint 算 Bone/Motion
        if self.split == 'train':
            # 随机平移
            if random.random() < 0.5:
                data = random_shift(data)
            # 随机旋转 (Sequence-level)
            if self.random_rot_flag:
                data = random_rot(data)

        # ===============================
        # 4. 模态转换 (物理一致性核心)
        # ===============================
        if self.modality == 'bone':
            bone_data = np.zeros_like(data)
            for v1, v2 in ntu_pairs:
                # v1, v2 是 1-based index
                bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
            
            # 【关键】NTU Bone 模态不应包含 Trajectory 或自环，直接替换
            data = bone_data
            
        elif self.modality == 'motion':
            # Motion = x(t+1) - x(t)
            motion = np.zeros_like(data)
            motion[:, :-1] = data[:, 1:] - data[:, :-1]
            # 最后一帧补0
            
            data = motion

        # ===============================
        # 5. Normalization
        # ===============================
        # 对 Bone/Motion 做 Z-Score 归一化，帮助模型收敛
        if self.modality in ['bone', 'motion']:
            std = data.std() + 1e-6
            data = data / std
        else:
            # Joint 模态处理：减去 Spine 中心，并把 Trajectory 放到特定通道 (保留原有逻辑)
            # 这种 Trajectory 技巧仅适用于 Joint
            traj = data[:, :, 20].copy()
            data = data - data[:, :, 20:21]
            data[:, :, 20] = traj

        # ===============================
        # 6. 单人/双人处理 (取 M=0)
        # ===============================
        # (C,T,V,M) -> (C,T,V)
        # 建议：如果显存允许，可以考虑 sum 或 max pooling，但目前取第一个人是最稳妥的复现方案
        data = data[:, :, :, 0]

        return data.astype(np.float32), label, index