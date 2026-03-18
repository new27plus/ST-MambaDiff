import torch
import time
import numpy as np
from thop import profile
from models.har_classifier import HAR_Classifier_Model

# ==========================================
# ⚙️ 配置区 (Configuration)
# ==========================================
# 必须与你 Phase 2 训练时的配置完全一致
NUM_CLASSES = 60       # NTU-60 (或者 120，对 GFLOPs 影响微乎其微)
NUM_JOINTS = 25        # NTU 关节数
MAX_FRAMES = 64        # 你训练时用的帧数
MODEL_CHANNELS = 128   # 你的 128 通道配置
MAMBA_D_STATE = 16     # Mamba 状态维度
DROPOUT = 0.0          # 测效率时 Dropout 不影响计算量，设为 0 即可

# FPS 测试配置
BATCH_SIZE = 64        # 模拟真实推理时的 Batch Size (通常越大 FPS 越高)
TEST_ITERATIONS = 100  # 跑多少次取平均

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"🚀 Starting Efficiency Test...")
    print(f"   Config: Ch={MODEL_CHANNELS}, Frames={MAX_FRAMES}, Batch={BATCH_SIZE}")

    # 1. 初始化模型 (Phase 2 模型)
    # 注意：这里不需要加载权重文件 (.pt)，因为权重的值不影响计算量和速度
    model = HAR_Classifier_Model(
        bone_pairs=[],
        num_classes=NUM_CLASSES,
        num_joints=NUM_JOINTS,
        in_channels=3,
        max_frames=MAX_FRAMES,
        model_channels=MODEL_CHANNELS,
        mamba_d_state=MAMBA_D_STATE,
        dropout=DROPOUT
    ).to(DEVICE)
    
    model.eval()

    # 2. 创建假数据 (Dummy Input)
    # Shape: (Batch, Channels, Frames, Joints)
    # 这里的 Batch 设为 1 用于计算单样本的 FLOPs
    dummy_input_1 = torch.randn(1, 3, MAX_FRAMES, NUM_JOINTS).to(DEVICE)
    
    # 3. 计算 Params 和 FLOPs
    print("\n[1] Calculating Params & GFLOPs...")
    # profile 会自动分析模型的每一层
    flops, params = profile(model, inputs=(dummy_input_1, ), verbose=False)
    
    print(f"-------------------------------------------")
    print(f"🧩 Parameters : {params / 1e6:.2f} M (Million)")
    print(f"🧮 GFLOPs     : {flops / 1e9:.2f} G (Giga-FLOPs per sample)")
    print(f"-------------------------------------------")

    # 4. 计算 FPS (Inference Speed)
    print("\n[2] Calculating Inference Speed (FPS)...")
    
    # 使用大 Batch 的假数据来测吞吐量
    dummy_input_batch = torch.randn(BATCH_SIZE, 3, MAX_FRAMES, NUM_JOINTS).to(DEVICE)
    
    # 预热 (Warmup): 让 GPU 进入工作状态
    print("   > Warming up GPU...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input_batch)
    
    # 正式计时
    print(f"   > Running {TEST_ITERATIONS} iterations...")
    torch.cuda.synchronize() # 确保之前的操作已完成
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(TEST_ITERATIONS):
            _ = model(dummy_input_batch)
    
    torch.cuda.synchronize() # 等待所有 GPU 任务完成
    end_time = time.time()
    
    total_time = end_time - start_time
    total_samples = BATCH_SIZE * TEST_ITERATIONS
    fps = total_samples / total_time
    
    print(f"-------------------------------------------")
    print(f"⏱️  Total Time : {total_time:.4f} s")
    print(f"🏎️  Throughput : {fps:.2f} FPS (Frames/Samples Per Second)")
    print(f"-------------------------------------------")

if __name__ == '__main__':
    main()