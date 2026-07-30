"""Cross-platform runtime inference for the crop-residue SMP U-Net."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn.functional as F

from model import make_model


MODEL_FILENAME: Final = "smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
MODEL_SHA256: Final = (
    "1dc9d28756769791cb9b80ad33aec9ecda74c35a8ad002d9bcf34fb808505840"
)
MODEL_STRIDE: Final = 32
PROJECT_ROOT: Final = Path(__file__).resolve().parent

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_MODEL_LOAD_COUNT = 0


class InferenceError(RuntimeError):
    """An inference failure with a message suitable for the Chinese UI."""


@dataclass(frozen=True)
class PredictionResult:
    mask: np.ndarray
    coverage: float
    image_size: tuple[int, int]


def default_model_path() -> Path:
    """Resolve the model without relying on the process working directory."""
    override = os.environ.get("RESIDUE_COVERAGE_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / MODEL_FILENAME


DEFAULT_CHECKPOINT = default_model_path()


def choose_device(requested: str = "auto") -> torch.device:
    """Choose CUDA, then Apple MPS, then CPU for automatic selection."""
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "mps" and not mps_available:
        raise InferenceError("当前环境无法使用 MPS，请改用 --device cpu 或自动选择。")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise InferenceError("当前环境无法使用 CUDA。")
        return torch.device("cuda")
    if requested != "cpu":
        raise InferenceError(f"不支持的运行设备：{requested}")
    return torch.device("cpu")


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise InferenceError(f"无法读取模型文件：{path}。{error}") from error
    return digest.hexdigest()


def load_image(path: str | Path) -> tuple[np.ndarray, torch.Tensor]:
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise InferenceError(f"图片文件不存在：{image_path}")

    try:
        with Image.open(image_path) as opened_image:
            rgb_image = ImageOps.exif_transpose(opened_image).convert("RGB")
            image_array = np.asarray(rgb_image).copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InferenceError(
            f"无法读取图片“{image_path.name}”。请上传有效的 JPG、JPEG、PNG、BMP、"
            "TIFF 或 WebP 图片。"
        ) from error

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise InferenceError("图片转换为 RGB 后格式仍然无效。")
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise InferenceError("图片尺寸无效，宽度和高度必须大于 0。")

    try:
        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
        )
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    except BaseException as error:
        if _is_out_of_memory(error):
            raise InferenceError("处理图片时内存不足，请关闭其他应用或使用更小的图片。") from error
        raise InferenceError(f"图片预处理失败：{error}") from error

    return image_array, image_tensor


def pad_to_model_stride(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    pad_height = (-height) % MODEL_STRIDE
    pad_width = (-width) % MODEL_STRIDE
    padded = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
    return padded, height, width


def load_model(
    checkpoint: str | Path,
    device: torch.device,
    *,
    expected_sha256: str | None = None,
) -> torch.nn.Module:
    global _MODEL_LOAD_COUNT

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise InferenceError(
            "未找到模型文件，暂时无法执行预测。\n"
            f"请将默认模型放到项目根目录：{MODEL_FILENAME}\n"
            "然后重新运行启动脚本。也可以通过 RESIDUE_COVERAGE_MODEL_PATH "
            f"指定模型位置。\n当前查找位置：{checkpoint_path}"
        )

    if expected_sha256:
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise InferenceError(
                "模型文件 SHA-256 校验失败，文件可能不完整或已损坏。"
                "请重新下载模型后再运行。\n"
                f"期望 SHA-256：{expected_sha256}\n实际 SHA-256：{actual_sha256}"
            )

    try:
        model = make_model(encoder_weights=None).to(device)
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        model.eval()
    except BaseException as error:
        if _is_out_of_memory(error):
            raise InferenceError(
                "加载模型时内存不足，请关闭其他应用后重新启动 ResidueCoverage。"
            ) from error
        raise InferenceError(
            "模型加载失败。请确认使用 Python 3.12 和 requirements-runtime.txt "
            f"中的依赖版本，并确认模型文件完整。\n技术信息：{error}"
        ) from error

    _MODEL_LOAD_COUNT += 1
    return model


def predict_mask(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise InferenceError("预测阈值必须在 0 到 1 之间。")

    try:
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
    except BaseException as error:
        if isinstance(error, InferenceError):
            raise
        if _is_out_of_memory(error):
            raise InferenceError(
                "推理时内存不足，请关闭其他应用或使用分辨率更小的图片后重试。"
            ) from error
        raise InferenceError(f"模型推理失败：{error}") from error


class ResiduePredictor:
    """Own one loaded model and reuse it for every prediction."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        requested_device: str = "auto",
        *,
        verify_checksum: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else default_model_path()
        self.device = choose_device(requested_device)
        expected = (
            MODEL_SHA256
            if verify_checksum and self.checkpoint.resolve() == default_model_path().resolve()
            else None
        )
        self.model = load_model(
            self.checkpoint,
            self.device,
            expected_sha256=expected,
        )

    def predict(self, image_path: str | Path, threshold: float = 0.5) -> PredictionResult:
        image, image_tensor = load_image(image_path)
        mask = predict_mask(self.model, image_tensor, self.device, threshold)
        coverage = float(mask.mean() * 100.0)
        height, width = image.shape[:2]
        return PredictionResult(
            mask=mask,
            coverage=coverage,
            image_size=(height, width),
        )


@lru_cache(maxsize=1)
def get_default_predictor() -> ResiduePredictor:
    return ResiduePredictor()


def model_load_count() -> int:
    return _MODEL_LOAD_COUNT
