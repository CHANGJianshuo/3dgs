# 3DGS 项目进度记录

## 环境信息
- 服务器: AutoDL A100 40GB PCIe (region-9), Ubuntu 22.04
- 存储: 30GB 系统盘 + 50GB 数据盘 (/root/autodl-tmp)
- 框架: gsplat 1.5.3 + simple_trainer (官方训练流程)
- 数据集: Mip-NeRF 360 全部 9 个场景 (15GB on disk)

## ✅ Step 1: 环境准备 (无卡模式) — 完成
- [x] conda env: gsplat (Python 3.10.20)
- [x] PyTorch 2.4.1 + CUDA 12.1
- [x] gsplat 1.5.3
- [x] 所有依赖 (pycolmap, viser, tyro, torchmetrics 1.4.3, opencv, ...)
- [x] fused_ssim stub (避免 CUDA 编译，纯 PyTorch SSIM 替代)
- [x] gsplat 仓库 (v1.5.3 tag, 含 simple_trainer.py)
- [x] Mip-NeRF 360 全 9 场景下载并解压
- [x] simple_trainer.py --help 验证通过
- [x] GitHub 仓库配置 (CHANGJianshuo/3dgs)

## 遇到的问题与解决
| 问题 | 解决方案 |
|------|---------|
| INRIA 数据源 (repo-sam.inria.fr) 国内不可达 | 改用 Google Cloud Storage (Jon Barron 官方源) |
| simple-knn 编译失败 | 不需要 - 我们用 gsplat 替代了原版 diff-rasterizer |
| fused-ssim 需要 CUDA 编译 (无卡模式没法编译) | 写 fused_ssim stub 用纯 PyTorch SSIM 替代 |
| GitHub clone fused-ssim 网络不稳 | stub 方案彻底绕过 |
| torch 2.3.1 + 新 torchmetrics 循环 import bug | 升级 torch 到 2.4.1 + 干净重装 (删残留文件) |
| torchvision 残留文件冲突 | rm -rf torch torchvision 目录后重装 |

## ⏸ Step 2: 训练 (等待开 GPU)
- [ ] 开 GPU 后验证 CUDA 可用
- [ ] 启动 garden 全分辨率训练 (30000 iter)
- [ ] 监控显存 + PSNR
- [ ] 训练完成后导出 .ply 模型

## 训练命令
```bash
bash /root/autodl-tmp/train_garden.sh
```

详见 train_garden.sh
