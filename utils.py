import os
from pathlib import Path

import cv2
import torch
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "residue_background"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth"
)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def read_file_names(directories):
    all_imgs = []

    for directory in directories:
        full_path = DATA_ROOT / directory
        file_names = [
            str(full_path / f[:-4]) for f in os.listdir(full_path)
            if f.lower().endswith(".jpg")
        ]
        all_imgs.extend(file_names)
        
    return sorted(all_imgs)

def jpg_read(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Can't read img: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = torch.from_numpy(img_rgb)
    image = image.permute(2,0,1) #(h,w,3) to (3,h,w)
    image = image.float()/255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def denormalise_image(image):
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image * std + mean).clamp(0, 1)

def tif_read(path):
    mask = tifffile.imread(path)
    if mask is None:
        raise ValueError(f"Can't read mask: {path}")
    
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0) #(h,w) to (1,h,w)
    mask = (mask>0).float()
    return mask

def visualise(image, real_mask, pred_mask):
    import matplotlib.pyplot as plt

    image_to_show = (
        denormalise_image(image)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    real_mask_to_show = (
        real_mask
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
    )

    pred_mask_to_show = (
        pred_mask
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
    )

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_to_show)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(
        real_mask_to_show,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(
        pred_mask_to_show,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
