import cv2
import numpy as np
from pathlib import Path

def detect_ground(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 农田颜色条件（hsv）
    lower_sky = np.array([0,20,0])
    upper_sky = np.array([35,255,255])

    # 农田识别 (255=农田 0=其他)
    mask = cv2.inRange(img_hsv, lower_sky, upper_sky)

    kernel = np.ones((5,5),dtype=np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    mask_bool_ground = mask_clean >0
    mask_bool_sky = mask_clean == 0
    mask_clean = np.stack([mask_clean,mask_clean,mask_clean], axis = -1)

    #只显示农田/只显示非农田
    sky_only = np.ones_like(img)*127
    ground_only = np.ones_like(img)*127
    sky_only[mask_bool_sky] = img[mask_bool_sky]
    ground_only[mask_bool_ground] = img[mask_bool_ground]

    # 组合图片
    result1 = np.concatenate([border(img),border(mask_clean)],axis = 1)
    result2 = np.concatenate([border(sky_only),border(ground_only)], axis = 1)
    result = np.concatenate([result1,result2],axis = 0)
    return result, mask_clean

# 图片加边框
def border(img):
    result = img
    px = 3
    shape = img.shape
    top = np.full((px,shape[1],3),255)
    side = np.full((shape[0]+px,px,3),255)
    bottom = np.full((px,shape[1]+2*px,3),255)
    
    result = np.concatenate([top,result],axis = 0)
    result = np.concatenate([side,result],axis = 1)
    result = np.concatenate([result,side],axis = 1)
    result = np.concatenate([result,bottom],axis = 0)
    return result


# 设置图片路径
img_dir = Path("photos")
img_paths = []
img_paths.extend(img_dir.glob("*.png")) # png only
result_dir = Path("results")
mask_dir = Path("masks")
result_dir.mkdir(exist_ok=True)
mask_dir.mkdir(exist_ok=True)

# 读取、识别、存储结果
count = 0
for img_path in img_paths:
    img_bgr = cv2.imread(str(img_path))
    result, mask = detect_ground(img_bgr)
    stem = img_path.stem

    count = count + 1
    print(f"识别了{count} 张图片")

    result_save_path = result_dir / f"{stem}.png"
    mask_save_path = mask_dir / f"{stem}.png"
    cv2.imwrite(str(result_save_path), result)
    cv2.imwrite(str(mask_save_path),mask)