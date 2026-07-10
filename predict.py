import joblib
import utils
import training
import numpy as np
import argparse

def predict(dirs):
    img_paths, mask_paths = utils.read_paths(dirs)

    if len(img_paths) != len(mask_paths):
        raise ValueError("图片和mask数量不匹配")

    bundle = joblib.load("residue_rf_model.joblib")
    clf = bundle["model"]
    feature_config = bundle["features"]

    saved_args = argparse.Namespace(**feature_config)

    img_num = 0
    X = training.make_features(saved_args,utils.jpg_read(img_paths[img_num]))
    y = utils.tiff_read(mask_paths[img_num]).reshape(-1).astype(np.uint8)
    pred = clf.predict(X)

    training.print_segmentation_metrics(y, pred)

    mask = utils.tiff_read(mask_paths[img_num])
    pred_mask = pred.reshape(mask.shape)
    result = np.stack([pred_mask*200,mask*200,np.zeros_like(pred_mask)], axis=-1)
    utils.show_plt(np.concatenate([utils.jpg_read(img_paths[img_num]),result],axis=1))

if __name__ == "__main__":
    dirs = ["residue_background/Zak-W-winterBarley_1m_20220401/IMG_0939",]
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    predict(dirs)