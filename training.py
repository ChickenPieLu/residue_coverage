import os
import tifffile
import cv2
import numpy as np
from pathlib import Path

# 路径读取
raw_dir = Path("residue_background/Limbaugh1-1m20220328/raw")
mask_dir = Path("residue_background/Limbaugh1-1m20220328/mask")
raw_paths, mask_paths = [],[]
raw_paths.extend(raw_dir.glob("*.jpg"))
mask_paths.extend(mask_dir.glob("*.tif"))

print(raw_paths[0], mask_paths[0])

def valid_mask(mask):
    return not np.any(mask[mask!=0]-255)

def tiff_read(path):
    mask = tifffile.imread(path)
    return mask

for mask_path in mask_paths:
    if not valid_mask(tiff_read(mask_path)):
        raise ValueError(f"invalid mask:{str(mask_path)}")

print("all masks are valid.")
