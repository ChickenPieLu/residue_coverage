"""Evaluate all three residue models on location E and create reports."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from legacy.classical_ml import utils as rf_utils
from legacy.classical_ml.evaluation import (
    load_model_bundle,
    predict_mask as predict_rf_mask,
)
from legacy.unet import utils as mini_utils
from legacy.unet.model import MiniUNet
from model import make_model
from training import utils as smp_utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs"
RF_THRESHOLD = 0.60
DEEP_THRESHOLD = 0.50

MODEL_CONFIGS = {
    "random_forest": {
        "name": "Random Forest",
        "checkpoint": rf_utils.DEFAULT_MODEL_PATH,
        "threshold": RF_THRESHOLD,
        "figure": "Figure_RF_MAE.png",
    },
    "mini_unet": {
        "name": "Mini U-Net",
        "checkpoint": mini_utils.DEFAULT_CHECKPOINT,
        "threshold": DEEP_THRESHOLD,
        "figure": "Figure_MiniUNet_MAE.png",
    },
    "smp_unet_resnet34": {
        "name": "SMP U-Net (ResNet-34)",
        "checkpoint": smp_utils.DEFAULT_CHECKPOINT,
        "threshold": DEEP_THRESHOLD,
        "figure": "Figure_smpUnet_MAE.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Random Forest, Mini U-Net, and SMP U-Net on location E."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics and report figures (default: logs).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used by the two neural networks (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Neural-network inference batch size (default: 4).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="Models to evaluate (default: all three).",
    )
    parser.add_argument(
        "--smp-checkpoint",
        type=Path,
        help="Override the SMP U-Net checkpoint path.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def load_pairs() -> list[tuple[Path, Path]]:
    image_paths, mask_paths = rf_utils.read_paths(
        rf_utils.location_dirs(["E"])
    )
    pairs = [
        (Path(image_path), Path(mask_path))
        for image_path, mask_path in zip(image_paths, mask_paths)
    ]
    if not pairs:
        raise ValueError("Location E contains no image/mask pairs.")
    return pairs


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def calculate_image_metrics(
    model_id: str,
    image_path: Path,
    mask_path: Path,
    prediction: np.ndarray,
) -> dict:
    truth = rf_utils.tiff_read(mask_path).astype(bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Prediction and mask shapes differ for {image_path.name}: "
            f"{prediction.shape} vs {truth.shape}"
        )

    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    true_pixels = int(truth.sum())
    predicted_pixels = int(prediction.sum())
    pixels = int(truth.size)
    true_negative = pixels - union
    false_positive = predicted_pixels - intersection
    false_negative = true_pixels - intersection

    true_coverage = true_pixels / pixels
    predicted_coverage = predicted_pixels / pixels
    coverage_error = predicted_coverage - true_coverage

    return {
        "model_id": model_id,
        "image": image_path.stem,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "intersection": intersection,
        "union": union,
        "true_pixels": true_pixels,
        "predicted_pixels": predicted_pixels,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "pixels": pixels,
        "iou": intersection / (union + 1e-8),
        "dice": 2 * intersection / (
            predicted_pixels + true_pixels + 1e-8
        ),
        "precision": intersection / (predicted_pixels + 1e-8),
        "recall": intersection / (true_pixels + 1e-8),
        "true_coverage": true_coverage,
        "predicted_coverage": predicted_coverage,
        "coverage_error": coverage_error,
        "absolute_coverage_error": abs(coverage_error),
    }


def aggregate_metrics(
    model_id: str,
    rows: list[dict],
    device: str,
) -> dict:
    config = MODEL_CONFIGS[model_id]
    intersection = sum(row["intersection"] for row in rows)
    union = sum(row["union"] for row in rows)
    true_pixels = sum(row["true_pixels"] for row in rows)
    predicted_pixels = sum(row["predicted_pixels"] for row in rows)
    true_negative = sum(row["true_negative"] for row in rows)
    false_positive = sum(row["false_positive"] for row in rows)
    pixels = sum(row["pixels"] for row in rows)

    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    dices = np.asarray([row["dice"] for row in rows], dtype=np.float64)
    errors = np.asarray(
        [row["coverage_error"] for row in rows], dtype=np.float64
    )

    return {
        "model_id": model_id,
        "model": config["name"],
        "checkpoint": str(config["checkpoint"]),
        "threshold": config["threshold"],
        "device": device,
        "images": len(rows),
        "pixel_iou": intersection / (union + 1e-8),
        "mean_image_iou": float(ious.mean()),
        "image_iou_std": float(ious.std()),
        "global_dice": 2 * intersection / (
            predicted_pixels + true_pixels + 1e-8
        ),
        "mean_image_dice": float(dices.mean()),
        "image_dice_std": float(dices.std()),
        "precision": intersection / (predicted_pixels + 1e-8),
        "recall": intersection / (true_pixels + 1e-8),
        "specificity": true_negative / (
            true_negative + false_positive + 1e-8
        ),
        "pixel_accuracy": (
            intersection + true_negative
        ) / (pixels + 1e-8),
        "true_coverage": true_pixels / pixels,
        "predicted_coverage": predicted_pixels / pixels,
        "coverage_mae": float(np.abs(errors).mean()),
        "coverage_rmse": float(np.sqrt(np.square(errors).mean())),
        "coverage_bias": float(errors.mean()),
        "maximum_coverage_error": float(np.abs(errors).max()),
    }


def print_progress(model_name: str, completed: int, total: int) -> None:
    if completed == total or completed % 16 == 0:
        print(f"{model_name}: {completed}/{total}", flush=True)


def evaluate_random_forest(
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    model_id = "random_forest"
    config = MODEL_CONFIGS[model_id]
    print(f"\nEvaluating {config['name']}...", flush=True)
    bundle = load_model_bundle(config["checkpoint"])
    rows = []

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        image = rf_utils.jpg_read(str(image_path))
        prediction = predict_rf_mask(
            bundle,
            image,
            threshold=config["threshold"],
        )
        rows.append(
            calculate_image_metrics(
                model_id,
                image_path,
                mask_path,
                prediction,
            )
        )
        print_progress(config["name"], index, len(pairs))

    metrics = aggregate_metrics(model_id, rows, device="CPU")
    selected = select_representative_rows(rows, metrics["coverage_mae"])
    add_selected_cases(metrics, selected)

    def predictor(path: Path) -> np.ndarray:
        return predict_rf_mask(
            bundle,
            rf_utils.jpg_read(str(path)),
            threshold=config["threshold"],
        )

    create_report_figure(
        config["name"],
        selected,
        predictor,
        output_dir / config["figure"],
    )
    del bundle
    gc.collect()
    return metrics, rows


def evaluate_torch_model(
    model_id: str,
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    config = MODEL_CONFIGS[model_id]
    print(f"\nEvaluating {config['name']}...", flush=True)

    if model_id == "mini_unet":
        model = MiniUNet().to(device)
        preprocess = mini_utils.jpg_read
    else:
        model = make_model(encoder_weights=None).to(device)
        preprocess = smp_utils.jpg_read

    state_dict = load_checkpoint(config["checkpoint"], device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    rows = []

    with torch.inference_mode():
        for offset in range(0, len(pairs), batch_size):
            batch_pairs = pairs[offset : offset + batch_size]
            images = torch.stack(
                [preprocess(str(image_path)) for image_path, _ in batch_pairs]
            ).to(device)
            predictions = (
                torch.sigmoid(model(images)) > config["threshold"]
            ).squeeze(1).cpu().numpy()

            for (image_path, mask_path), prediction in zip(
                batch_pairs, predictions
            ):
                rows.append(
                    calculate_image_metrics(
                        model_id,
                        image_path,
                        mask_path,
                        prediction,
                    )
                )
            print_progress(
                config["name"],
                min(offset + batch_size, len(pairs)),
                len(pairs),
            )

    metrics = aggregate_metrics(model_id, rows, device=str(device))
    selected = select_representative_rows(rows, metrics["coverage_mae"])
    add_selected_cases(metrics, selected)

    def predictor(path: Path) -> np.ndarray:
        image = preprocess(str(path)).unsqueeze(0).to(device)
        with torch.inference_mode():
            return (
                torch.sigmoid(model(image))[0, 0] > config["threshold"]
            ).cpu().numpy()

    create_report_figure(
        config["name"],
        selected,
        predictor,
        output_dir / config["figure"],
    )
    del model, state_dict
    gc.collect()
    return metrics, rows


def select_representative_rows(
    rows: list[dict],
    coverage_mae: float,
) -> list[tuple[str, dict]]:
    ordered = sorted(rows, key=lambda row: row["absolute_coverage_error"])
    best = ordered[0]
    worst = ordered[-1]
    middle_candidates = [
        row for row in rows if row is not best and row is not worst
    ]
    average = min(
        middle_candidates,
        key=lambda row: abs(
            row["absolute_coverage_error"] - coverage_mae
        ),
    )
    return [
        ("Best (lowest coverage error)", best),
        ("Typical (closest to mean MAE)", average),
        ("Worst (highest coverage error)", worst),
    ]


def add_selected_cases(
    metrics: dict,
    selected: list[tuple[str, dict]],
) -> None:
    for key, (_, row) in zip(("best", "typical", "worst"), selected):
        metrics[f"{key}_case"] = row["image"]
        metrics[f"{key}_case_absolute_error"] = row[
            "absolute_coverage_error"
        ]


def create_report_figure(
    model_name: str,
    selected: list[tuple[str, dict]],
    predictor: Callable[[Path], np.ndarray],
    output_path: Path,
) -> None:
    error_cmap = ListedColormap(["black", "red", "deepskyblue"])
    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 12),
        constrained_layout=True,
    )

    for row_index, (case_label, row) in enumerate(selected):
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        image = rf_utils.jpg_read(str(image_path))
        truth = rf_utils.tiff_read(mask_path).astype(bool)
        prediction = np.asarray(predictor(image_path), dtype=bool)

        error_map = np.zeros(truth.shape, dtype=np.uint8)
        error_map[np.logical_and(prediction, np.logical_not(truth))] = 1
        error_map[np.logical_and(np.logical_not(prediction), truth)] = 2

        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(
            prediction, cmap="gray", vmin=0, vmax=1
        )
        axes[row_index, 3].imshow(
            error_map, cmap=error_cmap, vmin=0, vmax=2
        )

        axes[row_index, 0].set_title(
            f"{case_label}\n{row['image']}\nIoU = {row['iou']:.4f}"
        )
        axes[row_index, 1].set_title(
            "Ground truth\n"
            f"Coverage = {row['true_coverage']:.2%}"
        )
        axes[row_index, 2].set_title(
            "Prediction\n"
            f"Coverage = {row['predicted_coverage']:.2%}"
        )
        axes[row_index, 3].set_title(
            "Error map\n"
            f"Absolute coverage error = "
            f"{row['absolute_coverage_error']:.2%}"
        )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        f"{model_name}: Representative Results on Test Location E\n"
        "Red = false positive, blue = false negative, black = correct",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure: {output_path}", flush=True)


def write_reports(
    output_dir: Path,
    summaries: list[dict],
    per_image_rows: list[dict],
) -> None:
    summary_csv = output_dir / "model_test_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    per_image_csv = output_dir / "model_test_per_image.csv"
    per_image_fields = [
        "model_id",
        "image",
        "iou",
        "dice",
        "precision",
        "recall",
        "true_coverage",
        "predicted_coverage",
        "coverage_error",
        "absolute_coverage_error",
    ]
    with per_image_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(
            {
                field: row[field]
                for field in per_image_fields
            }
            for row in per_image_rows
        )

    report_json = output_dir / "model_test_metrics.json"
    report_json.write_text(
        json.dumps(
            {
                "dataset": "residue_background/E",
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report_txt = output_dir / "model_test_report.txt"
    lines = [
        "Three-model evaluation on residue_background/E",
        f"Images: {summaries[0]['images']}",
        "",
    ]
    for metrics in summaries:
        lines.extend(
            [
                metrics["model"],
                f"  Checkpoint: {metrics['checkpoint']}",
                f"  Threshold: {metrics['threshold']:.2f}",
                f"  Pixel IoU: {metrics['pixel_iou']:.4f}",
                f"  Mean per-image IoU: {metrics['mean_image_iou']:.4f}",
                f"  Global Dice: {metrics['global_dice']:.4f}",
                (
                    "  Mean per-image Dice: "
                    f"{metrics['mean_image_dice']:.4f}"
                ),
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall: {metrics['recall']:.4f}",
                f"  Specificity: {metrics['specificity']:.4f}",
                f"  Pixel accuracy: {metrics['pixel_accuracy']:.4f}",
                f"  True coverage: {metrics['true_coverage']:.2%}",
                (
                    "  Predicted coverage: "
                    f"{metrics['predicted_coverage']:.2%}"
                ),
                f"  Coverage MAE: {metrics['coverage_mae']:.2%}",
                f"  Coverage RMSE: {metrics['coverage_rmse']:.2%}",
                f"  Coverage bias: {metrics['coverage_bias']:+.2%}",
                (
                    "  Maximum coverage error: "
                    f"{metrics['maximum_coverage_error']:.2%}"
                ),
                (
                    "  Best case: "
                    f"{metrics['best_case']} "
                    f"({metrics['best_case_absolute_error']:.2%})"
                ),
                (
                    "  Typical case (closest to mean MAE): "
                    f"{metrics['typical_case']} "
                    f"({metrics['typical_case_absolute_error']:.2%})"
                ),
                (
                    "  Worst case: "
                    f"{metrics['worst_case']} "
                    f"({metrics['worst_case_absolute_error']:.2%})"
                ),
                "",
            ]
        )
    report_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved summary: {summary_csv}", flush=True)
    print(f"Saved per-image data: {per_image_csv}", flush=True)
    print(f"Saved JSON: {report_json}", flush=True)
    print(f"Saved text report: {report_txt}", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smp_checkpoint:
        MODEL_CONFIGS["smp_unet_resnet34"]["checkpoint"] = (
            args.smp_checkpoint.expanduser().resolve()
        )
    device = choose_device(args.device)
    pairs = load_pairs()
    print(
        f"Dataset: residue_background/E ({len(pairs)} image/mask pairs)",
        flush=True,
    )
    print(f"Neural-network device: {device}", flush=True)

    summaries = []
    per_image_rows = []

    if "random_forest" in args.models:
        metrics, rows = evaluate_random_forest(pairs, output_dir)
        summaries.append(metrics)
        per_image_rows.extend(rows)

    for model_id in ("mini_unet", "smp_unet_resnet34"):
        if model_id not in args.models:
            continue
        metrics, rows = evaluate_torch_model(
            model_id,
            pairs,
            device,
            args.batch_size,
            output_dir,
        )
        summaries.append(metrics)
        per_image_rows.extend(rows)

    write_reports(output_dir, summaries, per_image_rows)


if __name__ == "__main__":
    main()
