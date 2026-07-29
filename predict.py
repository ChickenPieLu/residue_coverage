from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from inference import (
    DEFAULT_CHECKPOINT,
    InferenceError,
    choose_device,
    load_image,
    load_model,
    predict_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict crop-residue coverage for one image and create a "
            "side-by-side image/mask figure."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT.name}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output figure path. Defaults to "
            "<input_name>_coverage.png beside the input image."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Residue probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the Matplotlib window after saving.",
    )
    return parser.parse_args()


def save_result_figure(
    image: np.ndarray,
    mask: np.ndarray,
    coverage: float,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Predicted mask\nResidue coverage: {coverage:.2f}%")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    image_path = args.image.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else image_path.with_name(f"{image_path.stem}_coverage.png")
    )

    try:
        device = choose_device(args.device)
        image, image_tensor = load_image(image_path)
        model = load_model(checkpoint_path, device)
        mask = predict_mask(model, image_tensor, device, args.threshold)
        coverage = float(mask.mean() * 100.0)
        save_result_figure(image, mask, coverage, output_path, args.show)
    except InferenceError as error:
        raise SystemExit(f"错误：{error}") from error

    print(f"Device: {device}")
    print(f"Residue coverage: {coverage:.2f}%")
    print(f"Result figure: {output_path}")

if __name__ == "__main__":
    main()
