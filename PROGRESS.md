# 3DGS 项目进度记录

## 环境信息
- 服务器: AutoDL A100 40GB PCIe (region-9)
- 系统: Ubuntu 22.04
- 存储: 30GB 系统盘 + 50GB 数据盘 (/root/autodl-tmp)
- 框架: gsplat + 原版 3DGS 训练流程
- 数据集: Mip-NeRF 360 garden (全分辨率)

## Step 1: 环境准备 (无卡模式)
- [ ] conda 环境创建 (Python 3.10 + PyTorch)
- [ ] 安装 gsplat 及依赖
- [ ] 克隆 3DGS 代码
- [ ] 下载 Mip-NeRF 360 garden 数据集
- [ ] 配置 GitHub 仓库

## Step 2: 训练 (开 GPU 后)
- [ ] 验证 GPU 可用
- [ ] 启动 garden 全分辨率训练
- [ ] 监控显存和训练状态
