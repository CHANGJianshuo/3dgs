# Step 2: GPU 验证

**时间**: 2026-04-10
**状态**: ✅ 完成

## GPU 信息
```
NVIDIA A100-PCIE-40GB
显存: 40960 MiB (可用 40327 MiB)
驱动: 550.90.07
计算能力: 8.0 (sm_80)
CUDA toolkit: 12.4
PyTorch CUDA: 12.1
```

## 验证步骤

### 1. nvidia-smi 检查
GPU 上线，40327 MiB 可用显存。

### 2. PyTorch CUDA 检查
```python
import torch
torch.cuda.is_available()  # True
torch.cuda.get_device_name(0)  # NVIDIA A100-PCIE-40GB
```

### 3. gsplat CUDA kernel 编译 + 渲染冒烟测试
- 首次 import gsplat 时 JIT 编译 CUDA kernel: **耗时 106.65 秒**
- 编译后渲染 100 个高斯到 100×100 图像: **成功**
- 输出形状: torch.Size([1, 100, 100, 3]) ✅

## 注意事项
- gsplat 第一次 import 编译耗时 ~2 分钟，之后会缓存复用
- TORCH_CUDA_ARCH_LIST 未设置时会编译所有架构，建议训练前设为 `8.0`（A100）以加速编译
