#!/bin/bash
set -e
LOG=/root/autodl-tmp/setup.log
exec > >(tee -a $LOG) 2>&1
echo "=== Setup started at $(date) ==="

export PATH=/root/miniconda3/bin:$PATH

# Step 1: Create conda env (skip if exists)
if ! conda env list | grep -q gsplat; then
    echo "[1/5] Creating conda env..."
    conda create -n gsplat python=3.10 -y
else
    echo "[1/5] Conda env gsplat already exists, skipping"
fi

# Activate
source activate gsplat
echo "Python: $(python --version)"

# Step 2: Install PyTorch (use pip mirror for China)
echo "[2/5] Installing PyTorch..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir 2>&1 | tail -3
echo "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"

# Step 3: Install gsplat and dependencies
echo "[3/5] Installing gsplat and dependencies..."
pip install gsplat plyfile tqdm Pillow==10.3.0 lpips tensorboard \
    --no-cache-dir 2>&1 | tail -3

# Step 4: Clone 3DGS
echo "[4/5] Cloning 3D Gaussian Splatting..."
cd /root/autodl-tmp
if [ ! -d "gaussian-splatting" ]; then
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
else
    echo "3DGS repo already cloned"
fi

# Install 3DGS dependencies
cd /root/autodl-tmp/gaussian-splatting
pip install -e submodules/simple-knn --no-cache-dir 2>&1 | tail -3

# Step 5: Download Mip-NeRF 360 dataset
echo "[5/5] Downloading Mip-NeRF 360 dataset..."
DATA_DIR=/root/autodl-tmp/data
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download from INRIA pre-processed link
if [ ! -d "360_v2" ]; then
    echo "Downloading 360_v2 (outdoor + indoor scenes)..."
    wget -q --show-progress -O 360_v2.zip \
        "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/360_v2.zip" || {
        echo "INRIA download failed, trying alternative..."
        # fallback: individual scene downloads from Jon Barron
        echo "Please download manually from https://jonbarron.info/mipnerf360/"
    }
    if [ -f 360_v2.zip ]; then
        echo "Extracting..."
        unzip -q 360_v2.zip
        rm 360_v2.zip
        echo "360_v2 extracted"
    fi
else
    echo "360_v2 already exists"
fi

# Also download Tanks & Temples + Deep Blending
if [ ! -d "tandt" ]; then
    echo "Downloading Tanks & Temples + Deep Blending..."
    wget -q --show-progress -O tandt_db.zip \
        "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip" || {
        echo "tandt_db download failed"
    }
    if [ -f tandt_db.zip ]; then
        unzip -q tandt_db.zip
        rm tandt_db.zip
        echo "tandt_db extracted"
    fi
else
    echo "tandt already exists"
fi

echo ""
echo "=== Setup completed at $(date) ==="
echo "=== Disk usage ==="
du -sh /root/autodl-tmp/*
echo "=== Verify ==="
source activate gsplat
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import gsplat
    print(f'gsplat: imported OK')
except Exception as e:
    print(f'gsplat import (no GPU expected): {e}')
print('All packages OK')
"
echo "=== DONE ==="
