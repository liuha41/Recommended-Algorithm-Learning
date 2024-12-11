import torch

# 检查 GPU 是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {is_cuda_available}")


if is_cuda_available:
    # 获取 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")

    # 获取每个 GPU 的名称
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i} 名称: {gpu_name}")

    # 获取当前 GPU 的索引
    current_device = torch.cuda.current_device()
    print(f"当前使用的 GPU 索引: {current_device}")

    # 获取当前 GPU 的名称
    current_gpu_name = torch.cuda.get_device_name(current_device)
    print(f"当前 GPU 名称: {current_gpu_name}")
else:
    print("没有可用的 GPU。")
