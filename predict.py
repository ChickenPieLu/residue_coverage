from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from model import make_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODEL_STRIDE = 32


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


def load_image(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    with Image.open(path) as opened_image:
        rgb_image = ImageOps.exif_transpose(opened_image).convert("RGB")
        image_array = np.asarray(rgb_image).copy()

    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return image_array, image_tensor


def pad_to_model_stride(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    pad_height = (-height) % MODEL_STRIDE
    pad_width = (-width) % MODEL_STRIDE
    return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate"), height, width


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")

    model = make_model(encoder_weights=None).to(device)
    try:
        state_dict = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    padded_image, original_height, original_width = pad_to_model_stride(image)

    with torch.inference_mode():
        logits = model(padded_image.to(device))
        probability = torch.sigmoid(logits)
        mask = probability > threshold

    return (
        mask[0, 0, :original_height, :original_width]
        .detach()
        .cpu()
        .numpy()
    )


def save_result_figure(
    image: np.ndarray,
    mask: np.ndarray,
    coverage: float,
    output_path: Path,
    show: bool,
) -> None:
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
    #figure.savefig(output_path, dpi=150, bbox_inches="tight")
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

    device = choose_device(args.device)
    image, image_tensor = load_image(image_path)
    model = load_model(checkpoint_path, device)
    mask = predict_mask(model, image_tensor, device, args.threshold)
    coverage = float(mask.mean() * 100.0)
    save_result_figure(image, mask, coverage, output_path, args.show)

    print(f"Device: {device}")
    print(f"Residue coverage: {coverage:.2f}%")
    # print(f"Result figure: {output_path}")



if __name__ == "__main__":
    main()
