import os
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

# 路径读取
def read_paths(raw_dir,mask_dir):
    raw_paths = [
        (raw_dir + f)for f in os.listdir(raw_dir)
        if f.lower().endswith(".jpg")
    ]
    mask_paths = [
        (mask_dir + f) for f in os.listdir(mask_dir)
        if f.lower().endswith(".tif")
    ]
    raw_paths, mask_paths = natsorted(raw_paths),natsorted(mask_paths) # 保证顺序
    return raw_paths, mask_paths

# 图片读取
def tiff_read(path):
    return tifffile.imread(path) #2D 只有明度

def jpg_read(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb #3D 彩色

# 确认mask只存在0,255
def valid_mask(mask):
    return not np.any(mask[mask!=0]-255)

def valid_masks(mask_paths):
    for mask_path in mask_paths:
        if not valid_mask(tiff_read(mask_path)):
            return False
    return True

# 红色mask
def masked_img(mask):
    empty = np.full(mask.shape, 0)
    mask1 = np.where(mask>0,120,0)
    coloured_mask = np.stack([mask1,empty,empty],axis = -1)
    return coloured_mask #3D 彩色

# 展示
def show_plt(img):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

raw_dir = "residue_background/Limbaugh1-1m20220328/raw/"
mask_dir = "residue_background/Limbaugh1-1m20220328/mask/"
raw_paths, mask_paths = read_paths(raw_dir,mask_dir)

img_num = 100
show_plt(np.clip((masked_img(tiff_read(mask_paths[img_num]))+jpg_read(raw_paths[img_num])),0,255))