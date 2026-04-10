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

## ✅ Step 2: GPU 验证 — 完成 (2026-04-10)
详见 docs/step2_gpu_verification.md

## ✅ Step 3: garden 全分辨率训练 — 完成 (2026-04-10)
- [x] 启动训练 (30000 iter, batch_size=1, full res)
- [x] 性能优化：CUDA fused-ssim (1.39 → 6.71 it/s, 4.8x)
- [x] 性能优化：DataLoader num_workers 4 → 16
- [x] 性能优化：图片预加载到 RAM 缓存 (~6.7 → ~12 it/s, 1.8x)
- [x] 训练跑完 30000 iter (2h03m, avg 4.04 it/s)
- [x] 监控显存 + PSNR
- [x] 从 ckpt 手动导出 .ply 模型 (默认 save_ply=False)

## 训练结果

| Step | PSNR | SSIM | LPIPS (alex) | num_GS | 渲染速度 |
|------|------|------|--------------|--------|----------|
| 7000 | 24.88 | 0.730 | 0.420 | 3.86M | 26 ms/img |
| **30000** | **26.10** | **0.786** | **0.267** | **6.38M** | 29 ms/img |

vs. 3DGS 论文 garden: PSNR 27.41 / SSIM 0.868 / LPIPS 0.103 (vgg)

差距来源：
- LPIPS 算法不同（我们 alex，论文 vgg），数值不直接可比
- gsplat 默认 strategy 与原版 3DGS 略有差异
- mid-eval 在 7000 跑了 ~10 分钟 PNG 写入 + 5 分钟 trajectory 视频，下次跑加 `--disable_video` + `--ply_steps 30000` + 减少 PNG 编码

## 输出文件 (服务器)
```
/root/autodl-tmp/output/garden_full/
├── ckpts/
│   ├── ckpt_6999_rank0.pt    (912 MB)
│   └── ckpt_29999_rank0.pt   (1.5 GB)
├── ply/
│   └── point_cloud_29999.ply (1.5 GB, 6.38M gaussians)
├── videos/
│   ├── traj_6999.mp4         (107 MB, 174 帧)
│   └── traj_29999.mp4        (134 MB, 174 帧)
├── stats/
│   ├── train_step{6999,29999}_rank0.json
│   └── val_step{6999,29999}.json
├── renders/                  (24 val PNGs each step)
└── tb/                       (tensorboard)
```

## ▶ Step 4: 传回本机 (进行中)
- [x] 拉回 val_step{6999,29999}.json
- [ ] 拉回 cfg.yml + train.log
- [ ] 拉回 point_cloud_29999.ply (1.5 GB)
- [ ] 拉回 traj_29999.mp4 (134 MB)

## 性能优化记录
| 阶段 | it/s | ETA | 说明 |
|------|------|-----|------|
| pure-PyTorch SSIM stub | 1.39 | ~6h | 训练能跑但太慢 |
| 装真正的 fused-ssim CUDA kernel | 6.71 | 1h12m | 关键瓶颈消除 |
| batch_size=4 (回退) | 1.20 | ~7h | CPU 数据加载成为新瓶颈 |
| num_workers=16 + image RAM cache | ~12.2 | ~41m | 161 张图预加载到 RAM (8.42 GB)，省去每 iter 的磁盘 I/O 和 cv2.remap |

## 服务器硬件 (实际)
- A100 PCIe 40GB ×1
- 80 CPU cores
- 629 GB RAM (596 GB 可用)

## 训练命令
```bash
bash /root/autodl-tmp/train_garden.sh
```

详见 train_garden.sh
