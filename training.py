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
    return tifffile.imread(path) > 0 # (h,w) 0 or 1

def jpg_read(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb # (h,w,c) 0-255

# 红色mask
def coloured_mask(mask):
    empty = np.full(mask.shape, 0)
    coloured_mask = np.stack([mask*120,empty,empty],axis = -1)
    return coloured_mask # (h,w,c) 0-255

# 展示
def show_plt(img):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# 每个像素的特征值 [R,G,B,H,S,V,X,Y], 0.0-1.0
def make_features(img_rgb):
    h, w, _ = img_rgb.shape
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    rgb = img_rgb.astype(np.float32)/ 255.0

    hsv = img_hsv.astype(np.float32)
    hsv[:,:,0] /= 179.0
    hsv[:,:,1] /= 255.0
    hsv[:,:,2] /= 255.0

    yy,xx = np.meshgrid(
        np.linspace(0,1,h),
        np.linspace(0,1,w),
        indexing = "ij"
    )
    xy = np.stack([xx,yy],axis = -1).astype(np.float32)

    features = np.concatenate([rgb,hsv,xy],axis = -1)
    return features.reshape(-1, features.shape[-1])

# 像素抽样(8000max)
def sample_pixels(img_path,mask_path,max_pixel=8000):
    img = jpg_read(img_path)
    mask = tiff_read(mask_path)

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(f"图像和mask大小不匹配: {img_path}")

    features = make_features(img)
    labels = mask.reshape(-1).astype(np.uint8)
    
    total_pixels = len(labels)
    if total_pixels > max_pixel:
        chosen = np.random.choice(total_pixels,size=max_pixel,replace=False)
    else:
        chosen = np.arange(total_pixels)
    
    return features[chosen],labels[chosen]

img_dir = "residue_background/Limbaugh1-1m20220328/raw/"
mask_dir = "residue_background/Limbaugh1-1m20220328/mask/"
img_paths, mask_paths = read_paths(img_dir,mask_dir)

x = 100
#show_plt(np.clip((coloured_mask(tiff_read(mask_paths[x]))+jpg_read(img_paths[x])),0,255))
print(sample_pixels(img_paths[x],mask_paths[x])[0][0,:])