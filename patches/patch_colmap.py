#!/usr/bin/env python
"""Patch gsplat colmap.py Dataset to preload all undistorted images into RAM cache."""
import re
import shutil
import sys

PATH = "/root/autodl-tmp/gsplat_repo/examples/datasets/colmap.py"
BACKUP = PATH + ".pre_cache_bak"

with open(PATH) as f:
    src = f.read()

if "_image_cache" in src:
    print("[patch] already patched, skipping")
    sys.exit(0)

shutil.copyfile(PATH, BACKUP)

# Patch 1: in Dataset.__init__, after self.indices assignment, build cache.
old_init_tail = (
    "        if split == \"train\":\n"
    "            self.indices = indices[indices % self.parser.test_every != 0]\n"
    "        else:\n"
    "            self.indices = indices[indices % self.parser.test_every == 0]\n"
)
new_init_tail = old_init_tail + (
    "\n"
    "        # === Preload all undistorted images into RAM cache (massive speedup) ===\n"
    "        print(f\"[Dataset] preloading {len(self.indices)} {split} images into RAM cache ...\", flush=True)\n"
    "        import time as _t\n"
    "        _t0 = _t.time()\n"
    "        self._image_cache = {}\n"
    "        self._mask_cache = {}\n"
    "        for _idx in self.indices:\n"
    "            _img = imageio.imread(self.parser.image_paths[_idx])[..., :3]\n"
    "            _cam_id = self.parser.camera_ids[_idx]\n"
    "            _params = self.parser.params_dict[_cam_id]\n"
    "            if len(_params) > 0:\n"
    "                _mapx = self.parser.mapx_dict[_cam_id]\n"
    "                _mapy = self.parser.mapy_dict[_cam_id]\n"
    "                _img = cv2.remap(_img, _mapx, _mapy, cv2.INTER_LINEAR)\n"
    "                _x, _y, _w, _h = self.parser.roi_undist_dict[_cam_id]\n"
    "                _img = _img[_y:_y + _h, _x:_x + _w]\n"
    "            self._image_cache[int(_idx)] = np.ascontiguousarray(_img)\n"
    "        _bytes = sum(v.nbytes for v in self._image_cache.values())\n"
    "        print(f\"[Dataset] preloaded {len(self._image_cache)} images, {_bytes/1e9:.2f} GB RAM, took {_t.time()-_t0:.1f}s\", flush=True)\n"
)

if old_init_tail not in src:
    print("[patch] ERROR: could not find init tail")
    sys.exit(1)
src = src.replace(old_init_tail, new_init_tail)

# Patch 2: replace __getitem__ image-loading + undistort block to use cache.
old_getitem_head = (
    "    def __getitem__(self, item: int) -> Dict[str, Any]:\n"
    "        index = self.indices[item]\n"
    "        image = imageio.imread(self.parser.image_paths[index])[..., :3]\n"
    "        camera_id = self.parser.camera_ids[index]\n"
    "        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K\n"
    "        params = self.parser.params_dict[camera_id]\n"
    "        camtoworlds = self.parser.camtoworlds[index]\n"
    "        mask = self.parser.mask_dict[camera_id]\n"
    "\n"
    "        if len(params) > 0:\n"
    "            # Images are distorted. Undistort them.\n"
    "            mapx, mapy = (\n"
    "                self.parser.mapx_dict[camera_id],\n"
    "                self.parser.mapy_dict[camera_id],\n"
    "            )\n"
    "            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)\n"
    "            x, y, w, h = self.parser.roi_undist_dict[camera_id]\n"
    "            image = image[y : y + h, x : x + w]\n"
)

new_getitem_head = (
    "    def __getitem__(self, item: int) -> Dict[str, Any]:\n"
    "        index = self.indices[item]\n"
    "        # === RAM cache hit: image is already undistorted+cropped ===\n"
    "        image = self._image_cache[int(index)]\n"
    "        camera_id = self.parser.camera_ids[index]\n"
    "        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K\n"
    "        params = self.parser.params_dict[camera_id]\n"
    "        camtoworlds = self.parser.camtoworlds[index]\n"
    "        mask = self.parser.mask_dict[camera_id]\n"
)

if old_getitem_head not in src:
    print("[patch] ERROR: could not find getitem block")
    sys.exit(1)
src = src.replace(old_getitem_head, new_getitem_head)

with open(PATH, "w") as f:
    f.write(src)
print(f"[patch] OK, backup at {BACKUP}")
