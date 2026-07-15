import joblib
import legacy.classical_ml.utils as utils
import legacy.classical_ml.training as training
import numpy as np
import argparse
import matplotlib.pyplot as plt

def error_map(mask, pred_mask):
    mask = mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    tp = mask & pred_mask
    fp = ~mask & pred_mask
    fn = mask & ~pred_mask

    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[tp] = [0, 200, 0]    # green
    result[fp] = [255, 0, 0]    # red
    result[fn] = [0, 0, 255]    # blue

    return result

def predict(dirs):
    img_paths, mask_paths = utils.read_paths(dirs)

    if len(img_paths) != len(mask_paths):
        raise ValueError("图片和mask数量不匹配")

    bundle = joblib.load("residue_rf_model.joblib")
    clf = bundle["model"]
    feature_config = bundle["features"]

    saved_args = argparse.Namespace(**feature_config)
    threshold = 0.60

    all_error = []
    all_signed_error = []
    all_true_coverage = []
    all_pred_coverage = []
    max_error_path = ""
    max_error_mask = None
    max_error_prob = None
    max_error = float('-inf')
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = utils.jpg_read(img_path)
        mask = utils.tiff_read(mask_path).astype(np.uint8)
        mask_1D = mask.reshape(-1)
        if not img.shape[:2] == mask.shape:
            raise ValueError("图片和mask大小不一致")

        features = training.make_features(saved_args, img)
        probability = clf.predict_proba(features)[:, 1]

        pred = probability >= threshold

        true_coverage = np.sum(mask_1D)/len(mask_1D)
        pred_coverage = np.sum(pred)/len(pred)
        error = np.abs(true_coverage-pred_coverage)
        signed_error = true_coverage-pred_coverage
        all_signed_error.append(signed_error)
        all_error.append(error)
        all_true_coverage.append(true_coverage)
        all_pred_coverage.append(pred_coverage)

        if error > max_error:
            max_error = error
            max_error_path = img_path
            max_error_mask = mask
            max_error_prob = probability.reshape(-1,512)

    print("threshold: "+str(threshold))
    print(f"mean error: {np.mean(all_error):.4f}")
    print(f"mean signed error: {np.mean(all_signed_error):.4f}")
    print(f"median error: {np.median(all_error):.4f}")
    print(f"highest error: {np.max(all_error):.4f}")
    print(f"lowest error: {np.min(all_error):.4f}")

    if max_error_path != "":
        max_name = max_error_path[21:-4]
        print(f"最大绝对误差图片：{max_name}")
        for t in [0.6]:
            max_error_pred = max_error_prob >= t
            e = np.abs(np.sum(max_error_pred)/max_error_pred.size - np.sum(max_error_mask)/max_error_mask.size)
            print(f"绝对误差：{e:.4f}")
            training.print_segmentation_metrics(max_error_mask,max_error_pred)
            #utils.show_plt(error_map(max_error_mask,max_error_pred))
    
    plt.scatter(all_true_coverage,all_pred_coverage,color='blue', marker='o')
    plt.axline((0, 0), slope=1)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",type=str,default="d")
    args = parser.parse_args()

    d = args.seq.upper()
    if (d not in ['A','B','C','D']) or len(d)!=1:
        raise ValueError(f"{d} 应该是A,B,C,D其中之一")
    dirs = ["residue_background/"+d,]
    predict(dirs)