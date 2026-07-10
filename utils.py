import os
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

# 路径读取
def read_paths(dirs):
    img_paths, mask_paths = [],[]
    for d in dirs:
        temp_img = [
            os.path.join(d, f)for f in os.listdir(d)
            if f.lower().endswith(".jpg")
        ]
        temp_mask = [
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith(".tif")
        ]
        img_paths.extend(natsorted(temp_img))
        mask_paths.extend(natsorted(temp_mask))
    return img_paths, mask_paths

# 图片读取
def tiff_read(path):
    return tifffile.imread(path) > 0 # (h,w) 0 or 1

def jpg_read(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb # (h,w,c) 0-255

# 测试用数组
def test_rgb_generator(size = (20,20)):
    return np.stack(
        [np.random.randint(255,size=size),
        np.random.randint(255,size=size),
        np.random.randint(255,size=size)
        ],
        axis = -1
        ).astype(np.uint8)

def test_gray_generator(size = (20,20)):
    return np.random.randint(255, size=size).astype(np.uint8)

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
    dir = ["residue_background/Zak-W-winterBarley_1m_20220401/IMG_0938",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0939",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0940",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0941",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0942",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0944",]
    _, mask_paths = read_paths(dir)
    for i in range(len(mask_paths)):
        img_path = mask_paths[i]
        mask = tifffile.imread(img_path)
        result = np.where(mask>0, 0, 255).astype(np.uint8)
        tifffile.imwrite(img_path,result,compression="lzw")

def time_convert(start, end):
    temp = end-start
    result = f"{int(temp//3600)}h "
    temp %= 3600
    result += f"{int(temp//60)}m "
    temp %= 60
    result += f"{temp:.2f}s"
    return result
