import numpy as np
import tifffile
from training import read_paths


raw_dir = "residue_background/Limbaugh1-1m20220328/raw/"
mask_dir = "residue_background/Limbaugh1-1m20220328/mask/"
raw_paths, mask_paths = read_paths(raw_dir,mask_dir)

# 黑底白特征<=>白底黑特征
# for i in range(80,161):
#     img_path = mask_paths[i]
#     mask = tifffile.imread(img_path)
#     result = np.where(mask>0, 0, 255).astype(np.uint8)
#     tifffile.imwrite(img_path,result,compression="lzw")