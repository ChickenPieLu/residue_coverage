import cv2
import numpy as np
import joblib
import utils #utils.py
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 每个像素的特征值 [R,G,B,H,S,V,n*n像素平均亮度], 0.0-1.0
def make_features(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    rgb = img_rgb.astype(np.float32)/ 255.0
    hsv = img_hsv.astype(np.float32)
    gray = hsv[:,:,-1].copy()

    hsv[:,:,0] /= 179.0
    hsv[:,:,1] /= 255.0
    hsv[:,:,2] /= 255.0

    local_mean = cv2.blur(gray,(5,5))[:,:,None]/255.0
    local_contrast = (gray - cv2.blur(gray,(5,5)))[:,:,None]/255.0

    feature_list = []
    if args.hsv: feature_list.append(hsv)
    if args.rgb: feature_list.append(rgb)
    if args.mean: feature_list.append(local_mean)
    if args.contrast: feature_list.append(local_contrast)

    features = np.concatenate(feature_list,axis = -1)
    return features.reshape(-1, features.shape[-1])

# 像素抽样(默认8000max)
def sample_pixels(img_path,mask_path,max_pixel=8000):
    img = utils.jpg_read(img_path)
    mask = utils.tiff_read(mask_path)

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
def main(args):
    seed = 114514
    np.random.seed(seed)
    dirs = [
        "residue_background/Limbaugh1-1m20220328",
        "residue_background/Ritzville2-SprWheat1m20220329",
        "residue_background/Ritzville3-WheatFallow1pass1m20220329"
        ]
    img_paths, mask_paths = utils.read_paths(dirs)

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
        max_depth=10,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1
    )

    print("开始训练")

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    end = time.perf_counter()

    print(f"训练完成 (时长 {utils.time_convert(start,end)} )，保存...")
    joblib.dump(clf, "residue_rf_model.joblib")

    print("训练结果：")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    ref_img = utils.jpg_read(test_img_dir[0]).astype(np.uint8)
    ref_mask = utils.tiff_read(test_mask_dir[0]).astype(np.uint8)
    ref_pred = clf.predict(make_features(ref_img)).reshape(512,512)
    black = np.zeros_like(ref_pred)
    top_right = np.stack([ref_pred*200,ref_mask*200,black],axis=-1)
    bot_right = np.stack([ref_pred*200,black,black],axis=-1)
    bot_left = np.stack([black,ref_mask*200,black],axis=-1)
    row_1 = np.concatenate([ref_img,top_right],axis=1)
    row_2 = np.concatenate([bot_left,bot_right],axis=1)
    utils.show_plt(np.concatenate([row_1,row_2],axis=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsv",action="store_true", default=False)
    parser.add_argument("--rgb",action="store_true", default=False)
    parser.add_argument("--mean",action="store_true", default=False)
    parser.add_argument("--contrast",action="store_true", default=False)

    args = parser.parse_args()
    main(args)