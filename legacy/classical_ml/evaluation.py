"""Shared Random Forest inference and evaluation helpers."""

from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np

from legacy.classical_ml import training, utils


def load_model_bundle(model_path=utils.DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到 Random Forest 模型：{model_path}\n"
            "请先运行 python -m legacy.classical_ml.training"
        )

    bundle = joblib.load(model_path)
    if not {"model", "features"} <= bundle.keys():
        raise ValueError(f"模型文件缺少 model/features 字段：{model_path}")
    return bundle


def predict_probability(bundle, image):
    feature_config = Namespace(**bundle["features"])
    features = training.make_features(feature_config, image)
    classifier = bundle["model"]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    return classifier.predict_proba(features)[:, positive_index]


def predict_mask(bundle, image, threshold=0.60):
    probability = predict_probability(bundle, image)
    return (probability >= threshold).reshape(image.shape[:2])


def evaluate_pairs(bundle, img_paths, mask_paths, threshold=0.60):
    if len(img_paths) != len(mask_paths):
        raise ValueError("图片和 mask 数量不匹配")
    if not img_paths:
        raise ValueError("测试集为空")

    per_image = []
    total_intersection = 0
    total_union = 0
    total_predicted = 0
    total_true = 0
    total_pixels = 0

    for img_path, mask_path in zip(img_paths, mask_paths):
        image = utils.jpg_read(img_path)
        mask = utils.tiff_read(mask_path).astype(bool)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"图片和 mask 大小不一致：{img_path}")

        prediction = predict_mask(bundle, image, threshold)
        intersection = int(np.logical_and(prediction, mask).sum())
        union = int(np.logical_or(prediction, mask).sum())
        true_count = int(mask.sum())
        pred_count = int(prediction.sum())
        pixels = mask.size

        true_coverage = true_count / pixels
        pred_coverage = pred_count / pixels
        coverage_error = pred_coverage - true_coverage

        per_image.append(
            {
                "name": utils.display_name(img_path),
                "iou": (intersection + 1e-7) / (union + 1e-7),
                "true_coverage": true_coverage,
                "pred_coverage": pred_coverage,
                "absolute_error": abs(coverage_error),
                "coverage_error": coverage_error,
            }
        )

        total_intersection += intersection
        total_union += union
        total_predicted += pred_count
        total_true += true_count
        total_pixels += pixels

    coverage_errors = np.asarray(
        [row["coverage_error"] for row in per_image], dtype=np.float64
    )
    image_ious = np.asarray(
        [row["iou"] for row in per_image], dtype=np.float64
    )

    return {
        "per_image": per_image,
        "pixel_iou": total_intersection / (total_union + 1e-8),
        "dice": 2 * total_intersection
        / (total_predicted + total_true + 1e-8),
        "precision": total_intersection / (total_predicted + 1e-8),
        "recall": total_intersection / (total_true + 1e-8),
        "true_ratio": total_true / total_pixels,
        "pred_ratio": total_predicted / total_pixels,
        "mean_iou": float(image_ious.mean()),
        "iou_std": float(image_ious.std()),
        "coverage_mae": float(np.abs(coverage_errors).mean()),
        "coverage_rmse": float(np.sqrt(np.square(coverage_errors).mean())),
        "coverage_bias": float(coverage_errors.mean()),
        "max_coverage_error": float(np.abs(coverage_errors).max()),
    }
