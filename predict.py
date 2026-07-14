import joblib
import utils
import training
import numpy as np
import argparse
from sklearn.metrics import classification_report

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
    threshold = 0.65

    all_error = []
    all_signed_error = []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = utils.jpg_read(img_path)
        mask = utils.tiff_read(mask_path).astype(np.uint8).reshape(-1)
        if not img.shape[:2] == mask.shape:
            raise ValueError("图片和mask大小不一致")

        features = training.make_features(saved_args, img)
        probability = clf.predict_proba(features)[:, 1]

        pred = probability >= threshold

        true_coverage = np.sum(mask)/len(mask)
        pred_coverage = np.sum(pred)/len(pred)
        error = np.abs(true_coverage-pred_coverage)
        signed_error = true_coverage-pred_coverage
        all_signed_error.append(signed_error)
        all_error.append(error)

    print("threshold: "+str(threshold))
    print(f"mean error: {np.mean(all_error):.4f}")
    print(f"mean signed error: {np.mean(all_signed_error):.4f}")
    print(f"median error: {np.median(all_error):.4f}")
    print(f"highest error: {np.max(all_error):.4f}")
    print(f"lowest error: {np.min(all_error):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",type=str,default="d")
    args = parser.parse_args()

    d = args.seq.upper()
    if (d not in ['A','B','C','D']) or len(d)!=1:
        raise ValueError(f"{d} 应该是A,B,C,D其中之一")
    dirs = ["residue_background/"+d,]
    predict(dirs)