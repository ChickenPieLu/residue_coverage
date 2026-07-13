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

def predict(dirs,args):
    img_paths, mask_paths = utils.read_paths(dirs)

    if len(img_paths) != len(mask_paths):
        raise ValueError("图片和mask数量不匹配")

    bundle = joblib.load("residue_rf_model.joblib")
    clf = bundle["model"]
    feature_config = bundle["features"]

    saved_args = argparse.Namespace(**feature_config)

    img_num = 20
    X = training.make_features(saved_args,utils.jpg_read(img_paths[img_num]))
    y = utils.tiff_read(mask_paths[img_num]).reshape(-1).astype(np.uint8)

    probability = clf.predict_proba(X)[:,1]
    pred = probability >= saved_args.prob/100.0

    print(classification_report(y,pred,zero_division=0))
    training.print_segmentation_metrics(y, pred)

    mask = utils.tiff_read(mask_paths[img_num])
    pred_mask = pred.reshape(mask.shape)
    errors = error_map(mask, pred_mask)

    if args.example:
        utils.show_plt(
            np.concatenate([
                utils.jpg_read(img_paths[img_num]),
                errors,
            ], axis=1)
        )

if __name__ == "__main__":
    dirs = ["residue_background/E",]
    parser = argparse.ArgumentParser()
    parser.add_argument("--example",action="store_true", default=False)
    args = parser.parse_args()
    predict(dirs,args)