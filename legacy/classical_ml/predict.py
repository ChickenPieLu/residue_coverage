"""Backward-compatible single-location Random Forest evaluation entry point."""

import argparse

import matplotlib.pyplot as plt

from legacy.classical_ml import utils
from legacy.classical_ml.evaluation import evaluate_pairs, load_model_bundle


def predict(locations, model_path=utils.DEFAULT_MODEL_PATH, threshold=0.60):
    img_paths, mask_paths = utils.read_paths(utils.location_dirs(locations))
    bundle = load_model_bundle(model_path)
    metrics = evaluate_pairs(bundle, img_paths, mask_paths, threshold)

    print(f"threshold: {threshold:.2f}")
    print(f"pixel IoU: {metrics['pixel_iou']:.4f}")
    print(f"mean per-image IoU: {metrics['mean_iou']:.4f}")
    print(f"coverage MAE: {metrics['coverage_mae']:.4f}")
    print(f"coverage RMSE: {metrics['coverage_rmse']:.4f}")
    print(f"coverage bias: {metrics['coverage_bias']:+.4f}")
    print(f"highest error: {metrics['max_coverage_error']:.4f}")

    true_coverage = [
        row["true_coverage"] for row in metrics["per_image"]
    ]
    pred_coverage = [
        row["pred_coverage"] for row in metrics["per_image"]
    ]
    plt.scatter(true_coverage, pred_coverage, color="blue", marker="o")
    plt.axline((0, 0), slope=1)
    plt.xlabel("True coverage")
    plt.ylabel("Predicted coverage")
    plt.show()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        choices=list("ABCDE"),
        default="E",
    )
    parser.add_argument("--model", default=utils.DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.60)
    arguments = parser.parse_args()
    predict(
        [arguments.location],
        model_path=arguments.model,
        threshold=arguments.threshold,
    )
