import os
import numpy as np
import json
import random
import math
from torch.utils.data import Dataset


class Feeder(Dataset):
    """
    Feeder for the NW-UCLA dataset. (Modified for robust multi-modal support)

    Arguments:
        data_path: Path to the root of the dataset containing .json skeleton files.
        label_path: Path to the JSON file containing data annotations.
        repeat: The number of times to repeat the dataset. Default: 1.
        modality: Type of data modality ('joint', 'bone', 'motion'). Default: 'joint'.
    """

    def __init__(self, data_path, label_path, repeat=1, modality="joint"):
        self.data_path = data_path
        self.label_path = label_path
        self.repeat = repeat
        self.modality = modality.lower()

        if self.modality not in ["joint", "bone", "motion", "bone-motion"]:
            raise ValueError(
                f"Unsupported modality: {self.modality}. Choose 'joint', 'bone', 'motion' or 'bone-motion'."
            )

        if "val" in self.label_path:
            self.train_val = "val"

        else:
            self.train_val = "train"

        # Fixed parameters for NW-UCLA

        self.num_joints = 20

        self.max_frames = 52

        self.bone_pairs = [
            (1 - 1, 2 - 1),
            (2 - 1, 3 - 1),
            (4 - 1, 3 - 1),
            (5 - 1, 3 - 1),
            (6 - 1, 5 - 1),
            (7 - 1, 6 - 1),
            (8 - 1, 7 - 1),
            (9 - 1, 3 - 1),
            (10 - 1, 9 - 1),
            (11 - 1, 10 - 1),
            (12 - 1, 11 - 1),
            (13 - 1, 1 - 1),
            (14 - 1, 13 - 1),
            (15 - 1, 14 - 1),
            (16 - 1, 15 - 1),
            (17 - 1, 1 - 1),
            (18 - 1, 17 - 1),
            (19 - 1, 18 - 1),
            (20 - 1, 19 - 1),
        ]

        # Re-map self.bone to use (child_idx, parent_idx) structure based on common kinematic trees
        # Head=2, Neck=1, Spine=0 (approximate, based on typical structures)
        # Assuming joint 2 (index 1) is Neck/Shoulder Center, joint 3 (index 2) is Head
        # Left Arm: 4(Sh) -> 5(Elb) -> 6(Wr) -> 7(Hand)
        # Right Arm: 8(Sh) -> 9(Elb) -> 10(Wr) -> 11(Hand)
        # Left Leg: 12(Hip) -> 13(Knee) -> 14(Ank) -> 15(Foot)
        # Right Leg: 16(Hip) -> 17(Knee) -> 18(Ank) -> 19(Foot)
        # Spine = 0, Base = 16? This mapping needs validation against UCLA definition.
        # Sticking to the user's `self.bone` interpretation (v1-child, v2-parent) for now, using 0-based index.

        self.bone = [
            (v1 - 1, v2 - 1)
            for v1, v2 in [
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 3),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 3),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
            ]
        ]

        self.load_data()

    def load_data(self):
        try:
            with open(self.label_path, "r") as f:
                self.data_dict = json.load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found at: {self.label_path}")

        self.data = []
        self.label = []

        print(f"Loading {self.train_val} data from {len(self.data_dict)} samples...")

        for item in self.data_dict:
            file_name = item["file_name"]
            full_skeleton_path = os.path.join(self.data_path, file_name + ".json")

            try:
                with open(full_skeleton_path, "r") as f:
                    json_file = json.load(f)

            except FileNotFoundError:
                print(
                    f"Warning: Skeleton file not found at '{full_skeleton_path}'. Skipping this sample."
                )
                continue

            except json.JSONDecodeError:
                print(
                    f"Warning: Error decoding JSON from '{full_skeleton_path}'. Skipping this sample."
                )
                continue

            # Extract skeleton data, assuming format might vary slightly
            if "skeletons" in json_file:
                skeletons = json_file["skeletons"]
            # Add checks for other possible keys if needed
            # elif 'skeleton' in json_file ...
            else:
                print(
                    f"Warning: Could not find 'skeletons' key in '{full_skeleton_path}'. Skipping."
                )
                continue

            # Ensure skeletons is a list or array
            if not isinstance(skeletons, (list, np.ndarray)):
                print(
                    f"Warning: 'skeletons' data in '{full_skeleton_path}' is not a list or array. Skipping."
                )
                continue

            try:
                # Convert to numpy array, assuming shape (T, V, C) or similar
                value = np.array(skeletons, dtype=np.float32)
                # Basic sanity check on dimensions (T >= 1, V=20, C=3)
                if (
                    value.ndim != 3
                    or value.shape[1] != self.num_joints
                    or value.shape[2] != 3
                    or value.shape[0] < 1
                ):
                    print(
                        f"Warning: Unexpected skeleton shape {value.shape} in '{full_skeleton_path}'. Expected (T, {self.num_joints}, 3). Skipping."
                    )
                    continue

            except Exception as e:
                print(
                    f"Warning: Error converting skeletons to numpy array in '{full_skeleton_path}': {e}. Skipping."
                )
                continue

            self.data.append(value)
            self.label.append(int(item["label"]) - 1)  # Convert label to 0-based index

    def __len__(self):

        return len(self.data) * self.repeat

    def rand_view_transform(self, X, agx, agy, s):
        """Applies random rotation and scaling. Input shape (T, V, C)."""
        agx = math.radians(agx)
        agy = math.radians(agy)

        # Rotation matrices
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(agx), -math.sin(agx)],
                [0, math.sin(agx), math.cos(agx)],
            ],
            dtype=np.float32,
        )

        Ry = np.array(
            [
                [math.cos(agy), 0, math.sin(agy)],
                [0, 1, 0],
                [-math.sin(agy), 0, math.cos(agy)],
            ],
            dtype=np.float32,
        )

        # Scaling matrix
        Ss = np.array([[s, 0, 0], [0, s, 0], [0, 0, s]], dtype=np.float32)

        # Combined transformation matrix
        transform_matrix = Ry @ Rx @ Ss  # Apply scaling last

        # Reshape X to (T*V, C), apply transform, reshape back
        T, V, C = X.shape
        X_flat = X.reshape(-1, C)
        X_transformed_flat = (
            X_flat @ transform_matrix.T
        )  # Use transpose for correct multiplication order
        X_transformed = X_transformed_flat.reshape(T, V, C)

        return X_transformed

    def __getitem__(self, index):

        true_index = index % len(self.data)
        label = self.label[true_index]
        # Load joint data: shape (T_original, V, C)
        value = self.data[true_index].copy()  # Ensure we work with a copy
        T_original, V, C = value.shape

        # 1. Centering (Subtract root joint position - e.g., joint index 2 which is hip center in many sets)
        # Or use the mean position of the first frame as an alternative
        # Using joint index 1 (Neck/Shoulder Center based on user's previous code)
        center = value[
            0, 1, :
        ]  # Center based on first frame, joint 1 (Neck/Shoulder Center?)
        value = value - center  # Broadcast subtraction

        # 2. Data Augmentation (Rotation and Scaling) - Only for training
        if self.train_val == "train":
            agx = random.uniform(-60, 60)  # Use uniform for angles
            agy = random.uniform(-60, 60)
            s = random.uniform(0.5, 1.5)
            value = self.rand_view_transform(value, agx, agy, s)
        # No augmentation for validation

        # 3. Normalization (Min-Max scaling to [-1, 1])
        # Flatten temporarily to find global min/max across all frames/joints
        value_flat = value.reshape(-1, C)
        min_vals = np.min(value_flat, axis=0)
        max_vals = np.max(value_flat, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = (
            1  # Avoid division by zero for static joints/dimensions
        )

        # Apply scaling
        scalerValue_flat = (value_flat - min_vals) / range_vals  # Scale to [0, 1]
        scalerValue_flat = scalerValue_flat * 2 - 1  # Scale to [-1, 1]
        # Reshape back to (T_original, V, C)
        scalerValue = scalerValue_flat.reshape(T_original, V, C)

        # 4. Temporal Sampling/Padding to self.max_frames
        data_processed = np.zeros((self.max_frames, V, C), dtype=np.float32)
        if T_original >= self.max_frames:
            if self.train_val == "train":
                # Random crop
                start_point = random.randint(0, T_original - self.max_frames)
                data_processed = scalerValue[
                    start_point : start_point + self.max_frames, :, :
                ]
            else:
                # Center crop or linspace sample for validation
                # Using linspace sampling for consistency
                indices = np.linspace(0, T_original - 1, self.max_frames, dtype=int)
                data_processed = scalerValue[indices, :, :]
        else:
            # Padding (repeat last frame or use zeros - using zeros here)
            data_processed[:T_original, :, :] = scalerValue
            # Alternative: Repeat last frame for padding
            # data_processed[T_original:, :, :] = scalerValue[-1:, :, :]

        # ----------------------------------------------------------
        # 5. [CORE MODIFICATION] Generate requested modality
        # This happens *after* spatial centering, augmentation, normalization,
        # and temporal processing.
        # ----------------------------------------------------------

        if self.modality == 'joint':
            final_data = data_processed # Shape (T, V, C)

        elif self.modality == 'bone':
            final_data = np.zeros_like(data_processed)
            for v1, v2 in self.bone: # Use 0-based bone pairs
                final_data[:, v1, :] = data_processed[:, v1, :] - data_processed[:, v2, :]
            # Bone 向量通常长度还可以，不需要剧烈放大，或者可以稍微放大一点比如 *1.0 (保持不变)

        elif self.modality == 'motion':
            final_data = np.zeros_like(data_processed)
            final_data[1:, :, :] = data_processed[1:, :, :] - data_processed[:-1, :, :]
            
            # 【专业修改】使用实例标准化 (Instance Normalization)
            # 计算当前样本的标准差 (加上一个极小值 eps 防止除以 0)
            std = final_data.std() + 1e-6
            # 强制将数据拉伸到单位方差，这样信号强度就和 Joint 一样了
            final_data = final_data / std 

        elif self.modality == 'bone-motion':
            bone_data = np.zeros_like(data_processed)
            for v1, v2 in self.bone:
                bone_data[:, v1, :] = data_processed[:, v1, :] - data_processed[:, v2, :]
            
            final_data = np.zeros_like(bone_data)
            final_data[1:, :, :] = bone_data[1:, :, :] - bone_data[:-1, :, :]
            
            # 【专业修改】同样对 Bone-Motion 做标准化
            std = final_data.std() + 1e-6
            final_data = final_data / std

        else:
             raise ValueError(f"Invalid modality '{self.modality}' encountered in getitem.")

        # 6. Final Transpose and Reshape for Model Input
        # Transpose to (C, T, V)
        final_data = np.transpose(final_data, (2, 0, 1))  # C, T, V
        # Add singleton dimension M (persons) -> (C, T, V, M=1)
        final_data = final_data[:, :, :, np.newaxis]

        return (
            final_data.astype(np.float32),
            label,
            index,
        )  # Return float32 numpy array, label, index
