#!/usr/bin/env python
"""Export PLY from a simple_trainer checkpoint."""
import sys
import torch
from gsplat import export_splats

CKPT = "/root/autodl-tmp/output/garden_full/ckpts/ckpt_29999_rank0.pt"
OUT  = "/root/autodl-tmp/output/garden_full/ply/point_cloud_29999.ply"

print(f"Loading {CKPT} ...")
ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
splats = ckpt["splats"]
print(f"Loaded, keys: {list(splats.keys())}")
print(f"num_GS = {splats['means'].shape[0]:,}")

# splats is an OrderedDict (state_dict), values are tensors
means    = splats["means"]
scales   = splats["scales"]
quats    = splats["quats"]
opacities = splats["opacities"]
sh0      = splats["sh0"]
shN      = splats["shN"]

print(f"Exporting PLY to {OUT} ...")
export_splats(
    means=means,
    scales=scales,
    quats=quats,
    opacities=opacities,
    sh0=sh0,
    shN=shN,
    format="ply",
    save_to=OUT,
)
print("Done.")
