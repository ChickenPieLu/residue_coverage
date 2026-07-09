import joblib
import utils
import training
import numpy as np
from sklearn.metrics import classification_report

def predict(dirs):
    raw_paths, mask_paths = utils.read_paths(dirs)

    img_num = 2
    X = training.make_features(utils.jpg_read(raw_paths[img_num]))
    y = utils.tiff_read(mask_paths[img_num]).reshape(-1).astype(np.uint8)

    clf = joblib.load("residue_rf_model.joblib")
    pred = clf.predict(X)

    print(classification_report(y,pred))

    pred_mask = pred.reshape(512,512)
    result = np.stack([pred_mask*200,utils.tiff_read(mask_paths[img_num])*200,np.zeros_like(pred_mask)], axis=-1)
    utils.show_plt(np.concatenate([utils.jpg_read(raw_paths[img_num]),result],axis=1))

if __name__ == "__main__":
    dirs = ["residue_background/Ritzville6-SprWheatWintPeas1m20220329/IMG_0889",]
    predict(dirs)