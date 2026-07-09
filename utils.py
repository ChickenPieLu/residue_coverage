import os
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

# 路径读取
def read_paths(raw_dir,mask_dir):
    raw_paths = [
        os.path.join(raw_dir, f)for f in os.listdir(raw_dir)
        if f.lower().endswith(".jpg")
    ]
    mask_paths = [
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
        if f.lower().endswith(".tif")
    ]
    raw_paths, mask_paths = natsorted(raw_paths),natsorted(mask_paths) # 保证顺序
    return raw_paths, mask_paths

# 图片读取
def tiff_read(path):
    return tifffile.imread(path) > 0 # (h,w) 0 or 1

def jpg_read(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb # (h,w,c) 0-255

# 红色mask
def coloured_mask(mask):
    empty = np.full(mask.shape, 0).astype(np.uint8)
    coloured_mask = np.stack([mask*120,empty,empty],axis = -1)
    return coloured_mask # (h,w,c) 0-255

# 展示
def show_plt(img):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# 黑底白特征<=>白底黑特征
def tiff_converter():
    raw_dir = "residue_background/Limbaugh1-1m20220328/raw/"
    mask_dir = "residue_background/Limbaugh1-1m20220328/mask/"
    _, mask_paths = read_paths(raw_dir,mask_dir)
    for i in range(80,161):
        img_path = mask_paths[i]
        mask = tifffile.imread(img_path)
        result = np.where(mask>0, 0, 255).astype(np.uint8)
        tifffile.imwrite(img_path,result,compression="lzw")