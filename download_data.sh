#!/bin/bash
set -e
LOG=/root/autodl-tmp/download.log
exec > >(tee -a $LOG) 2>&1
echo "=== Download started at $(date) ==="

DATA_DIR=/root/autodl-tmp/data
mkdir -p $DATA_DIR
cd $DATA_DIR

# Mip-NeRF 360 official source from Jon Barron (Google Cloud Storage)
# Includes: images, images_2/4/8 downsampled, COLMAP sparse poses
URL_MAIN="https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
URL_EXTRA="https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"

# Download main 7 scenes (~7GB)
if [ ! -f "360_v2.zip" ] || [ ! -s "360_v2.zip" ]; then
    echo "[1/2] Downloading 360_v2.zip (main 7 scenes, ~7GB)..."
    wget -c --tries=5 --timeout=60 -O 360_v2.zip "$URL_MAIN"
else
    echo "[1/2] 360_v2.zip already exists, size: $(du -h 360_v2.zip | cut -f1)"
fi

# Download extra 2 scenes (flowers, treehill) (~1GB)
if [ ! -f "360_extra_scenes.zip" ] || [ ! -s "360_extra_scenes.zip" ]; then
    echo "[2/2] Downloading 360_extra_scenes.zip (flowers, treehill, ~1GB)..."
    wget -c --tries=5 --timeout=60 -O 360_extra_scenes.zip "$URL_EXTRA"
else
    echo "[2/2] 360_extra_scenes.zip already exists, size: $(du -h 360_extra_scenes.zip | cut -f1)"
fi

# Extract
echo "=== Extracting ==="
unzip -q -o 360_v2.zip
unzip -q -o 360_extra_scenes.zip

# Cleanup zips to save disk
echo "=== Cleanup zips ==="
rm -f 360_v2.zip 360_extra_scenes.zip

# Verify
echo "=== Verify ==="
ls -la $DATA_DIR/
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
    if [ -d "$DATA_DIR/$scene" ]; then
        n_imgs=$(ls $DATA_DIR/$scene/images/ 2>/dev/null | wc -l)
        has_sparse=$([ -d "$DATA_DIR/$scene/sparse/0" ] && echo "yes" || echo "NO")
        echo "  $scene: $n_imgs images, sparse: $has_sparse"
    else
        echo "  $scene: MISSING"
    fi
done

echo ""
df -h /root/autodl-tmp
echo "=== Download done at $(date) ==="
