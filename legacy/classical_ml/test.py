"""Evaluate the archived Random Forest on the unseen location E."""

import argparse

from legacy.classical_ml import utils
from legacy.classical_ml.evaluation import evaluate_pairs, load_model_bundle


def test(model_path=utils.DEFAULT_MODEL_PATH, threshold=0.60):
    img_paths, mask_paths = utils.read_paths(utils.location_dirs(["E"]))
    bundle = load_model_bundle(model_path)

    print("testing set E:")
    print(f"threshold={threshold:.2f}, images={len(img_paths)}")
    metrics = evaluate_pairs(
        bundle,
        img_paths,
        mask_paths,
        threshold=threshold,
    )

    for row in metrics["per_image"]:
        print(
            f"{row['name']}: "
            f"IoU={row['iou']:.4f}, "
            f"true={row['true_coverage']:.4f}, "
            f"pred={row['pred_coverage']:.4f}, "
            f"error={row['absolute_error']:.4f}"
        )

    print("\nAggregate segmentation results on E:")
    print(
        f"IoU={metrics['pixel_iou']:.4f}, "
        f"Dice={metrics['dice']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"true_ratio={metrics['true_ratio']:.4f}, "
        f"pred_ratio={metrics['pred_ratio']:.4f}"
    )

    print("\nPer-image results on E:")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"IoU standard deviation: {metrics['iou_std']:.4f}")
    print(f"Coverage MAE: {metrics['coverage_mae']:.4f}")
    print(f"Coverage RMSE: {metrics['coverage_rmse']:.4f}")
    print(f"Coverage bias: {metrics['coverage_bias']:+.4f}")
    print(
        "Maximum coverage error: "
        f"{metrics['max_coverage_error']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=utils.DEFAULT_MODEL_PATH,
        help="Random Forest joblib 模型路径",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    arguments = parser.parse_args()
    test(arguments.model, arguments.threshold)
