"""
Mega-NeRF dataset Parser/Dataset for gsplat simple_trainer.

Drop-in replacement for examples/datasets/colmap.py:Parser. Reads the
Mega-NeRF on-disk format used by Mill19 / building-pixsfm:

    <data_dir>/
        train/rgbs/000000.jpg ...
        train/metadata/000000.pt ...   # dict with W, H, intrinsics, c2w
        val/rgbs/...
        val/metadata/...
        coordinates.pt                  # optional: origin_drb + pose_scale_factor

Each metadata .pt is a dict with keys:
    'W'          (int)         image width
    'H'          (int)         image height
    'intrinsics' (Tensor[4])   [fx, fy, cx, cy]
    'c2w'        (Tensor[3,4]) camera-to-world (Mega-NeRF DRB convention)

We DON'T have a sparse point cloud here, so self.points is empty and
training MUST use --init_type random.

Mega-NeRF c2w uses NeRF/OpenGL convention "RUB" (right-up-back), confirmed
by its ray formula  d = [(i-cx)/fx, -(j-cy)/fy, -1]. COLMAP / gsplat expect
OpenCV "RDF" (right-down-forward). Conversion: flip Y and Z columns of R.
    R_opencv = R_rub @ diag(1, -1, -1)
(translation t is unchanged because the camera origin is the same point.)

Usage in simple_trainer.py:
    from datasets.mega_nerf import Parser as MegaNerfParser
    if os.path.isdir(os.path.join(data_dir, 'train', 'metadata')):
        parser = MegaNerfParser(data_dir, factor=cfg.data_factor, ...)
    else:
        parser = ColmapParser(...)
"""
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


def _list_metadata(split_dir: str) -> List[str]:
    md_dir = os.path.join(split_dir, "metadata")
    if not os.path.isdir(md_dir):
        return []
    return sorted(
        [f for f in os.listdir(md_dir) if f.endswith(".pt")]
    )


class Parser:
    """Mega-NeRF parser. Mirrors the COLMAP Parser interface used by gsplat."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,  # ignored: Mega-NeRF has explicit train/val split
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every  # kept for API compat

        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        if not os.path.isdir(train_dir):
            raise ValueError(f"Mega-NeRF train dir not found: {train_dir}")

        train_files = _list_metadata(train_dir)
        val_files = _list_metadata(val_dir)
        if len(train_files) == 0:
            raise ValueError(f"No train metadata .pt files in {train_dir}/metadata")

        # Build merged list. Mega-NeRF .jpg names match .pt names by stem.
        # We use a path prefix ('train/' or 'val/') in image_names so they
        # are unique across splits.
        records = []  # (image_name, image_path, metadata_path, is_val)
        for f in train_files:
            stem = os.path.splitext(f)[0]
            records.append(
                (
                    f"train/{stem}",
                    os.path.join(train_dir, "rgbs", stem + ".jpg"),
                    os.path.join(train_dir, "metadata", f),
                    False,
                )
            )
        for f in val_files:
            stem = os.path.splitext(f)[0]
            records.append(
                (
                    f"val/{stem}",
                    os.path.join(val_dir, "rgbs", stem + ".jpg"),
                    os.path.join(val_dir, "metadata", f),
                    True,
                )
            )
        # We tag val records by index so Dataset can split them.
        self._is_val = np.array([r[3] for r in records], dtype=bool)

        image_names = [r[0] for r in records]
        image_paths = [r[1] for r in records]

        # Load all metadata .pt files (small dicts, fast)
        camtoworlds = []
        Ks_dict = {}
        params_dict = {}
        imsize_dict = {}
        mask_dict = {}
        camera_ids = []

        # Mega-NeRF RUB (Right,Up,Back) -> OpenCV RDF (Right,Down,Forward)
        # Flip Y and Z columns: R_opencv = R_rub @ diag(1, -1, -1)
        rub_to_rdf = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        for idx, (_name, _ipath, mpath, _isv) in enumerate(
            tqdm(records, desc="Loading Mega-NeRF metadata")
        ):
            md = torch.load(mpath, map_location="cpu", weights_only=False)
            W = int(md["W"]) // factor
            H = int(md["H"]) // factor
            intr = np.asarray(md["intrinsics"], dtype=np.float32).reshape(-1)
            fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
            fx /= factor
            fy /= factor
            cx /= factor
            cy /= factor
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            c2w = np.asarray(md["c2w"], dtype=np.float32)
            if c2w.shape == (3, 4):
                c2w_h = np.eye(4, dtype=np.float32)
                c2w_h[:3, :4] = c2w
            elif c2w.shape == (4, 4):
                c2w_h = c2w.astype(np.float32)
            else:
                raise ValueError(f"Unexpected c2w shape {c2w.shape} in {mpath}")

            # Convert axis convention: world stays the same, only the camera
            # local frame changes. R_new = R_old @ drb_to_rdf
            R_old = c2w_h[:3, :3]
            t = c2w_h[:3, 3]
            R_new = R_old @ rub_to_rdf
            c2w_new = np.eye(4, dtype=np.float32)
            c2w_new[:3, :3] = R_new
            c2w_new[:3, 3] = t
            camtoworlds.append(c2w_new)

            # Each image gets its own camera_id (allows per-image intrinsics)
            cam_id = idx
            camera_ids.append(cam_id)
            Ks_dict[cam_id] = K
            params_dict[cam_id] = np.empty(0, dtype=np.float32)  # already undistorted
            imsize_dict[cam_id] = (W, H)
            mask_dict[cam_id] = None

        camtoworlds = np.stack(camtoworlds, axis=0)

        # Optional: load coordinates.pt for normalization (origin + scale)
        # The Mega-NeRF coordinates.pt centers and scales the world such that
        # cameras lie roughly inside the unit cube. gsplat normalize=True does
        # something similar from the camera positions, so we just rely on that.
        coords_path = os.path.join(data_dir, "coordinates.pt")
        if os.path.exists(coords_path):
            coords = torch.load(coords_path, map_location="cpu", weights_only=False)
            origin = np.asarray(coords["origin_drb"], dtype=np.float32)
            pose_scale = float(coords["pose_scale_factor"])
            # Translate cameras: t' = (t - origin) / pose_scale
            camtoworlds[:, :3, 3] = (camtoworlds[:, :3, 3] - origin) / pose_scale
            print(
                f"[MegaNerfParser] applied coordinates.pt: origin={origin}, "
                f"pose_scale={pose_scale:.4f}"
            )

        # No SfM points: caller must use --init_type random
        points = np.zeros((0, 3), dtype=np.float32)
        points_err = np.zeros((0,), dtype=np.float32)
        points_rgb = np.zeros((0, 3), dtype=np.uint8)
        point_indices = {}  # empty; load_depths must be False

        transform = np.eye(4, dtype=np.float32)
        if normalize:
            from .normalize import similarity_from_cameras, transform_cameras
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            transform = T1

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform
        self.bounds = np.array([0.01, 1.0])
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}

        # No undistortion needed (already undistorted)
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}

        # Scene scale from camera positions
        cam_locs = camtoworlds[:, :3, 3]
        center = np.mean(cam_locs, axis=0)
        self.scene_scale = float(np.max(np.linalg.norm(cam_locs - center, axis=1)))
        print(
            f"[MegaNerfParser] {len(image_names)} images "
            f"({int((~self._is_val).sum())} train, {int(self._is_val.sum())} val), "
            f"scene_scale={self.scene_scale:.3f}"
        )

        # Load one image to validate sizes match metadata (some Mega-NeRF
        # releases store metadata for a different downsample factor than the
        # actual jpg). Adjust K + imsize accordingly.
        actual = imageio.imread(image_paths[0])[..., :3]
        ah, aw = actual.shape[:2]
        mw, mh = imsize_dict[camera_ids[0]]
        if (ah, aw) != (mh, mw):
            sx, sy = aw / mw, ah / mh
            for cid, K in self.Ks_dict.items():
                K[0, :] *= sx
                K[1, :] *= sy
                self.Ks_dict[cid] = K
                w, h = self.imsize_dict[cid]
                self.imsize_dict[cid] = (int(round(w * sx)), int(round(h * sy)))
            print(
                f"[MegaNerfParser] WARNING: actual image {(aw,ah)} != metadata {(mw,mh)}, "
                f"rescaled K by ({sx:.3f}, {sy:.3f})"
            )


class Dataset:
    """Mega-NeRF dataset. Splits via parser._is_val (explicit train/val)."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        if load_depths:
            raise ValueError("load_depths not supported for Mega-NeRF (no SfM points)")
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = False

        all_indices = np.arange(len(parser.image_names))
        if split == "train":
            self.indices = all_indices[~parser._is_val]
        else:
            self.indices = all_indices[parser._is_val]

        # Preload all images into RAM cache (massive speedup; same trick as
        # the patched colmap.Dataset)
        print(
            f"[MegaNerfDataset] preloading {len(self.indices)} {split} images into RAM cache ...",
            flush=True,
        )
        import time as _t
        _t0 = _t.time()
        self._image_cache = {}
        for _idx in self.indices:
            _img = imageio.imread(self.parser.image_paths[_idx])[..., :3]
            self._image_cache[int(_idx)] = np.ascontiguousarray(_img)
        _bytes = sum(v.nbytes for v in self._image_cache.values())
        print(
            f"[MegaNerfDataset] preloaded {len(self._image_cache)} images, "
            f"{_bytes/1e9:.2f} GB RAM, took {_t.time()-_t0:.1f}s",
            flush=True,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = int(self.indices[item])
        image = self._image_cache[index]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        return {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
