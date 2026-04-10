#!/bin/bash
# Train Mega-NeRF Building scene with gsplat simple_trainer.
# Uses the patched simple_trainer.py + datasets/mega_nerf.py for auto-detection.
#
# Building has 1940 train + 21 val photos at ~16 MP. With data_factor=4 the
# training resolution is ~1450x965, which fits comfortably in 40 GB.
#
# We use init_type=random because Mega-NeRF's release does NOT include a sparse
# point cloud — only refined poses. init_num_pts=500_000 gives the densifier
# something substantial to start from for a city-block scale scene.

set -euo pipefail
source /root/miniconda3/bin/activate gsplat

DATA=/root/autodl-tmp/data/building-pixsfm
OUT=/root/autodl-tmp/output/building_full
LOG=$OUT/train.log

mkdir -p "$OUT"

cd /root/autodl-tmp/gsplat_repo

python examples/simple_trainer.py default \
    --data_dir "$DATA" \
    --result_dir "$OUT" \
    --data_factor 4 \
    --batch_size 1 \
    --max_steps 30000 \
    --eval_steps 7000 30000 \
    --save_steps 7000 30000 \
    --init_type random \
    --init_num_pts 500000 \
    --init_extent 3.0 \
    --disable_viewer \
    --disable_video \
    2>&1 | tee "$LOG"
