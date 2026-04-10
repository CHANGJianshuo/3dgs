#!/usr/bin/env python
"""
Idempotent patch: make simple_trainer.py auto-detect Mega-NeRF datasets
and use datasets/mega_nerf.py:{Parser,Dataset} in place of the COLMAP ones.

Also installs datasets/mega_nerf.py from the same directory as this script
into the gsplat repo's examples/datasets/.

Usage:
    python patch_mega_nerf.py [--gsplat-repo /root/autodl-tmp/gsplat_repo]
"""
import argparse
import os
import shutil
import sys

MARKER = "# === MEGA-NERF AUTO-DETECT PATCH ==="

OLD_IMPORT = "from datasets.colmap import Dataset, Parser\n"
NEW_IMPORT = (
    "# === MEGA-NERF AUTO-DETECT PATCH ===\n"
    "from datasets.colmap import Dataset as _ColmapDataset, Parser as _ColmapParser\n"
    "from datasets.mega_nerf import Dataset as _MegaNerfDataset, Parser as _MegaNerfParser\n"
    "import os as _patch_os\n"
    "def _pick_parser_dataset(data_dir):\n"
    "    if _patch_os.path.isdir(_patch_os.path.join(data_dir, 'train', 'metadata')):\n"
    "        print('[patch] detected Mega-NeRF format dataset, using MegaNerfParser')\n"
    "        return _MegaNerfParser, _MegaNerfDataset\n"
    "    return _ColmapParser, _ColmapDataset\n"
    "Parser = _ColmapParser  # default for type annotations\n"
    "Dataset = _ColmapDataset\n"
)

OLD_PARSER_BLOCK = (
    "        self.parser = Parser(\n"
    "            data_dir=cfg.data_dir,\n"
    "            factor=cfg.data_factor,\n"
    "            normalize=cfg.normalize_world_space,\n"
    "            test_every=cfg.test_every,\n"
    "        )\n"
    "        self.trainset = Dataset(\n"
    "            self.parser,\n"
    "            split=\"train\",\n"
    "            patch_size=cfg.patch_size,\n"
    "            load_depths=cfg.depth_loss,\n"
    "        )\n"
    "        self.valset = Dataset(self.parser, split=\"val\")\n"
)

NEW_PARSER_BLOCK = (
    "        # === MEGA-NERF AUTO-DETECT PATCH ===\n"
    "        _ParserCls, _DatasetCls = _pick_parser_dataset(cfg.data_dir)\n"
    "        self.parser = _ParserCls(\n"
    "            data_dir=cfg.data_dir,\n"
    "            factor=cfg.data_factor,\n"
    "            normalize=cfg.normalize_world_space,\n"
    "            test_every=cfg.test_every,\n"
    "        )\n"
    "        self.trainset = _DatasetCls(\n"
    "            self.parser,\n"
    "            split=\"train\",\n"
    "            patch_size=cfg.patch_size,\n"
    "            load_depths=cfg.depth_loss,\n"
    "        )\n"
    "        self.valset = _DatasetCls(self.parser, split=\"val\")\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsplat-repo", default="/root/autodl-tmp/gsplat_repo")
    args = ap.parse_args()

    repo = args.gsplat_repo
    examples_ds = os.path.join(repo, "examples", "datasets")
    trainer = os.path.join(repo, "examples", "simple_trainer.py")
    src_mega = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mega_nerf.py")

    if not os.path.isfile(src_mega):
        print(f"ERROR: mega_nerf.py source not found at {src_mega}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(examples_ds):
        print(f"ERROR: gsplat datasets dir not found: {examples_ds}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(trainer):
        print(f"ERROR: simple_trainer.py not found: {trainer}", file=sys.stderr)
        sys.exit(1)

    # 1) Copy mega_nerf.py
    dst_mega = os.path.join(examples_ds, "mega_nerf.py")
    shutil.copyfile(src_mega, dst_mega)
    print(f"[patch] installed {dst_mega}")

    # 2) Patch simple_trainer.py (idempotent via MARKER)
    with open(trainer, "r") as f:
        content = f.read()

    if MARKER in content:
        print("[patch] simple_trainer.py already patched, refreshing block...")
        # Even if already patched, ensure the mega_nerf.py copy is fresh.
        # No further edits needed.
        return

    # backup
    bak = trainer + ".pre_meganerf_bak"
    if not os.path.exists(bak):
        shutil.copyfile(trainer, bak)
        print(f"[patch] backup -> {bak}")

    if OLD_IMPORT not in content:
        print(f"ERROR: expected import line not found:\n  {OLD_IMPORT!r}", file=sys.stderr)
        sys.exit(2)
    if OLD_PARSER_BLOCK not in content:
        print("ERROR: expected parser block not found in simple_trainer.py", file=sys.stderr)
        sys.exit(3)

    content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
    content = content.replace(OLD_PARSER_BLOCK, NEW_PARSER_BLOCK, 1)

    with open(trainer, "w") as f:
        f.write(content)
    print(f"[patch] simple_trainer.py patched")


if __name__ == "__main__":
    main()
