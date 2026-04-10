#!/bin/bash
# Train Mip-NeRF 360 garden scene at full resolution using gsplat
# Target hardware: A100 40GB PCIe
set -e

DATA=/root/autodl-tmp/data/garden
OUT=/root/autodl-tmp/output/garden_full
LOG=/root/autodl-tmp/output/garden_full.log

mkdir -p $(dirname $OUT)
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 sm_80
source /root/miniconda3/bin/activate gsplat

cd /root/autodl-tmp/gsplat_repo

# Training command - matches original 3DGS training (default strategy = vanilla)
# --data_factor 1 = full resolution (no downsampling)
# default strategy = original 3DGS adaptive density control
python examples/simple_trainer.py default \
    --data_dir $DATA \
    --result_dir $OUT \
    --data_factor 1 \
    --batch_size 1 \
    --max_steps 30000 \
    --eval_steps 7000 30000 \
    --save_steps 7000 30000 \
    --disable_viewer \
    --disable_video \
    2>&1 | tee $LOG
