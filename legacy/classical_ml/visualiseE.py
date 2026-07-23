"""Visualise the same three location-E cases used by the U-Net archive."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from legacy.classical_ml import utils
from legacy.classical_ml.evaluation import load_model_bundle, predict_mask


CASES = [
    {
        "name": "Successful case",
        "file": utils.DATA_ROOT / "E" / "IMG_0937_part06",
    },
    {
        "name": "Typical case",
        "file": utils.DATA_ROOT / "E" / "IMG_0944_part11",
    },
    {
        "name": "High-coverage underestimation",
        "file": utils.DATA_ROOT / "E" / "IMG_0946_part04",
    },
]


def visualise_e(
    model_path=utils.DEFAULT_MODEL_PATH,
    threshold=0.60,
    save_path=None,
    show=True,
):
    bundle = load_model_bundle(model_path)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    error_cmap = ListedColormap(["black", "red", "deepskyblue"])

    for row, case in enumerate(CASES):
        image = utils.jpg_read(str(case["file"]) + ".jpg")
        mask = utils.tiff_read(str(case["file"]) + ".tif").astype(bool)
        prediction = predict_mask(bundle, image, threshold)

        intersection = np.logical_and(prediction, mask).sum()
        union = np.logical_or(prediction, mask).sum()
        iou = (intersection + 1e-7) / (union + 1e-7)
        true_coverage = mask.mean()
        pred_coverage = prediction.mean()
        coverage_error = abs(pred_coverage - true_coverage)

        error_map = np.zeros(mask.shape, dtype=np.uint8)
        error_map[np.logical_and(prediction, ~mask)] = 1
        error_map[np.logical_and(~prediction, mask)] = 2

        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(
            case["name"], fontsize=12, fontweight="bold"
        )
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(error_map, cmap=error_cmap, vmin=0, vmax=2)

        axes[row, 0].set_title(
            f"{Path(case['file']).name}\nIoU = {iou:.4f}"
        )
        axes[row, 1].set_title(
            f"Ground truth\nCoverage = {true_coverage:.2%}"
        )
        axes[row, 2].set_title(
            f"Prediction\nCoverage = {pred_coverage:.2%}"
        )
        axes[row, 3].set_title(
            f"Errors\nAbsolute error = {coverage_error:.2%}"
        )

        print(
            f"{case['name']}: "
            f"IoU={iou:.4f}, "
            f"true={true_coverage:.4f}, "
            f"pred={pred_coverage:.4f}, "
            f"error={coverage_error:.4f}"
        )

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        "Random Forest Results on Test Location E\n"
        "Error map: red = false positive, blue = false negative",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=utils.DEFAULT_MODEL_PATH,
        help="Random Forest joblib 模型路径",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--save", help="可选的图片输出路径")
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args()
    visualise_e(
        model_path=arguments.model,
        threshold=arguments.threshold,
        save_path=arguments.save,
        show=not arguments.no_show,
    )
