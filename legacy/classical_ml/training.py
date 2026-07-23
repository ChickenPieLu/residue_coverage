import cv2
import numpy as np
import joblib
import legacy.classical_ml.utils as utils #utils.py
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
)

# 每个像素的特征值 [R,G,B,H,S,V,n*n像素平均亮度], 0.0-1.0
def make_features(args,img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    rgb = img_rgb.astype(np.float32)/ 255.0
    hsv = img_hsv.astype(np.float32)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)/255.0
    gray = hsv[:,:,-1].copy()
    gray_float = gray.astype(np.float32)/255.0

    hsv[:,:,0] /= 179.0
    hsv[:,:,1] /= 255.0
    hsv[:,:,2] /= 255.0

    area = (args.area,args.area)
    mean_raw = cv2.blur(gray,area)
    mean_squared_raw = cv2.blur(gray ** 2,area)
    variance_raw = np.maximum(mean_squared_raw - mean_raw ** 2,0)
    std_raw = np.sqrt(variance_raw)

    local_mean = mean_raw[:,:,None]/255.0
    local_contrast = (gray - mean_raw)[:,:,None]/255.0
    std = std_raw[:,:,None]/255.0

    feature_list = []
    if args.hsv: feature_list.append(hsv)
    if args.rgb: feature_list.append(rgb)
    if args.mean: feature_list.append(local_mean)
    if args.contrast: feature_list.append(local_contrast)
    if args.std: feature_list.append(std)
    if args.lab: feature_list.append(lab)

    if len(feature_list)==0: # 默认
        feature_list.append(hsv)
        feature_list.append(local_contrast)
        feature_list.append(std)

    features = np.concatenate(feature_list,axis = -1)
    return features.reshape(-1, features.shape[-1])

# 像素抽样(默认8000max)
def sample_pixels(args,img_path,mask_path,max_pixel=8000):
    img = utils.jpg_read(img_path)
    mask = utils.tiff_read(mask_path)

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(f"图像和mask大小不匹配: {img_path}")

    features = make_features(args,img)
    labels = mask.reshape(-1).astype(np.uint8)

    true_i = np.nonzero(labels == 1)[0]
    false_i = np.nonzero(labels == 0)[0]

    positive_target = max_pixel // (1 + args.neg)
    negative_target = max_pixel - positive_target

    positive_count = min(len(true_i), positive_target)
    negative_count = min(len(false_i), negative_target)

    true_chosen = np.random.choice(
        true_i,
        size=positive_count,
        replace=False
    )

    false_chosen = np.random.choice(
        false_i,
        size=negative_count,
        replace=False
    )

    chosen = np.concatenate([true_chosen, false_chosen])

    # 打乱顺序，不让前面全是正类、后面全是负类
    np.random.shuffle(chosen)

    return features[chosen], labels[chosen]

# 输出训练数据
def print_segmentation_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1).astype(np.uint8)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.uint8)

    print(f"IoU:       {jaccard_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Dice:      {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")

# 核心训练部分
def main(args):
    seed = 114514
    np.random.seed(seed)
    training_dirs, validation_dirs = utils.process_seq(args.seq)

    print("训练：")
    for d in training_dirs:
        print(d)
    print("验证：")
    for d in validation_dirs:
        print(d)
    print("训练用特征：")
    feature_list = []
    if args.hsv: feature_list.append("hsv")
    if args.rgb: feature_list.append("rgb")
    if args.mean: feature_list.append("local_mean")
    if args.contrast: feature_list.append("local_contrast")
    if args.std: feature_list.append("std")
    if args.lab: feature_list.append("lab")
    if len(feature_list) == 0: print("['hsv','local_contrast','std']")
    else: print(str(feature_list))

    train_img_dir, train_mask_dir = utils.read_paths(training_dirs)
    val_img_dir, val_mask_dir = utils.read_paths(validation_dirs)

    if len(train_img_dir) != len(train_mask_dir):
        raise ValueError("训练集图片和mask数量不匹配")
    if len(val_img_dir) != len(val_mask_dir):
        raise ValueError("验证集图片和mask数量不匹配")
    
    train_features, train_labels = [],[]
    for i in range(len(train_img_dir)):
        try:
            features, labels = sample_pixels(args,train_img_dir[i],train_mask_dir[i])
            train_features.append(features)
            train_labels.append(labels)
        except ValueError:
            print(f"错误: {str(train_img_dir[i])} 跳过该图片")
            continue

    X_train, y_train = np.concatenate(train_features), np.concatenate(train_labels)
    print("成功创建训练与验证集")

    class_counts = np.bincount(y_train, minlength=2)

    print("背景像素数量：", class_counts[0])
    print("秸秆像素数量：", class_counts[1])
    print("秸秆占比：", y_train.mean())

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        class_weight=None,
        random_state=seed,
        n_jobs=-1
    )

    print("开始训练\n")

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    training_end = time.perf_counter()

    print(f"训练完成 (时长 {utils.time_convert(start,training_end)} )，保存...")

    model_bundle = {
    "model": clf,
    "features": {
        "hsv": args.hsv,
        "rgb": args.rgb,
        "mean": args.mean,
        "contrast": args.contrast,
        "std": args.std,
        "lab": args.lab,
        "area":args.area,
        },
    }
    joblib.dump(model_bundle, args.model)

    print("训练结果：")
    all_true = []
    all_probability = []

    for img_path, mask_path in zip(val_img_dir, val_mask_dir):
        img = utils.jpg_read(img_path)
        mask = utils.tiff_read(mask_path).astype(np.uint8)

        features = make_features(args, img)
        probability = clf.predict_proba(features)[:, 1]

        all_true.append(mask.reshape(-1))
        all_probability.append(probability)

    all_true = np.concatenate(all_true)
    all_probability = np.concatenate(all_probability)

    th_list = [0.60]
    if args.threshold: th_list = [0.50, 0.55, 0.60, 0.65, 0.70]
    for threshold in th_list:
        print(f"(阈值：{str(threshold)})")
        all_pred = all_probability >= threshold

        print(classification_report(all_true, all_pred, zero_division=0))
        print_segmentation_metrics(all_true, all_pred)
    
    end = time.perf_counter()
    print(f"总用时：{utils.time_convert(start,end)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsv",action="store_true", default=False)
    parser.add_argument("--rgb",action="store_true", default=False)
    parser.add_argument("--mean",action="store_true", default=False)
    parser.add_argument("--contrast",action="store_true", default=False)
    parser.add_argument("--std",action="store_true",default=False)
    parser.add_argument("--example",action="store_true", default=False)
    parser.add_argument("--lab",action="store_true", default=False)
    parser.add_argument("--threshold",action="store_true", default=False)
    parser.add_argument("--seq",type=str,default="ABCD")
    parser.add_argument(
        "--model",
        default=utils.DEFAULT_MODEL_PATH,
        help="训练后模型的保存路径",
    )
    parser.add_argument("--area",type=int,choices=range(1,50),default=15)
    parser.add_argument(
        "--neg",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="背景像素相对于秸秆像素的抽样比例"
    )
    args = parser.parse_args()
    main(args)
