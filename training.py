import os
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# 每个像素的特征值 [R,G,B,H,S,V,X,Y], 0.0-1.0
def make_features(img_rgb):
    h, w, _ = img_rgb.shape
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    rgb = img_rgb.astype(np.float32)/ 255.0

    hsv = img_hsv.astype(np.float32)
    hsv[:,:,0] /= 179.0
    hsv[:,:,1] /= 255.0
    hsv[:,:,2] /= 255.0

    # yy,xx = np.meshgrid(
    #     np.linspace(0,1,h),
    #     np.linspace(0,1,w),
    #     indexing = "ij"
    # )
    # xy = np.stack([xx,yy],axis = -1).astype(np.float32)

    features = np.concatenate([rgb,hsv],axis = -1)
    return features.reshape(-1, features.shape[-1])

# 像素抽样(默认8000max)
def sample_pixels(img_path,mask_path,max_pixel=8000):
    img = jpg_read(img_path)
    mask = tiff_read(mask_path)

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(f"图像和mask大小不匹配: {img_path}")

    features = make_features(img)
    labels = mask.reshape(-1).astype(np.uint8)

    # 分层抽样(一半 true, 一半 false)
    true_i = np.nonzero(labels)[0]
    false_i = np.nonzero(labels == 0)[0]

    lim = int(max_pixel/2)
    if len(true_i) > lim:
        true_chosen = np.random.choice(true_i, size=lim, replace=False)
    else: true_chosen = true_i
    if len(false_i) > lim:
        false_chosen = np.random.choice(false_i, size = lim, replace=False)
    else: false_chosen = false_i

    chosen = np.concatenate([true_chosen,false_chosen])
    
    return features[chosen],labels[chosen]

# 核心训练部分
def main():
    seed = 114514
    np.random.seed(seed)
    img_dir = "residue_background/Limbaugh1-1m20220328/raw/"
    mask_dir = "residue_background/Limbaugh1-1m20220328/mask/"
    img_paths, mask_paths = read_paths(img_dir,mask_dir)

    if len(img_paths) != len(mask_paths):
        raise ValueError("图片和mask数量不匹配")
    
    # 分离训练和测试集
    train_img_dir, test_img_dir, train_mask_dir, test_mask_dir = train_test_split(img_paths,mask_paths,random_state=seed)
    
    train_features, train_labels = [],[]
    test_features, test_labels = [],[]
    for i in range(len(train_img_dir)):
        try:
            features, labels = sample_pixels(train_img_dir[i],train_mask_dir[i])
            train_features.append(features)
            train_labels.append(labels)
        except ValueError:
            print(f"错误: {str(train_img_dir[i])} 跳过该图片")
            continue
    for j in range(len(test_img_dir)):
        try:
            features, labels = sample_pixels(test_img_dir[j],test_mask_dir[j])
            test_features.append(features)
            test_labels.append(labels)
        except ValueError:
            print(f"错误: {str(test_img_dir[j])} 跳过该图片")
            continue
    X_train, y_train = np.concatenate(train_features), np.concatenate(train_labels)
    X_test, y_test = np.concatenate(test_features), np.concatenate(test_labels)

    print("成功创建训练与测试集")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1
    )

    print("开始训练")
    clf.fit(X_train, y_train)

    print("训练成功，测试")

    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    img,mask = jpg_read(test_img_dir[0]),tiff_read(test_mask_dir[0])
    pred = clf.predict(make_features(img))
    pred_2d = pred.reshape(512,512)
    result = np.stack([pred_2d*200,mask*200,np.zeros_like(pred_2d)],axis=-1)
    show_plt(result)

main()