import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from legacy.unet import utils
from legacy.unet.dataset import ResidueDataset
from legacy.unet.model import MiniUNet


def visualise_e():
    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Load final model
    model = MiniUNet().to(device)
    model.load_state_dict(
        torch.load(
            utils.DEFAULT_CHECKPOINT,
            map_location=device
        )
    )
    model.eval()

    # Three representative cases from location E
    cases = [
        {
            "name": "Successful case",
            "file": str(utils.DATA_ROOT / "E" / "IMG_0937_part06")
        },
        {
            "name": "Typical case",
            "file": str(utils.DATA_ROOT / "E" / "IMG_0944_part11")
        },
        {
            "name": "High-coverage underestimation",
            "file": str(utils.DATA_ROOT / "E" / "IMG_0946_part04")
        }
    ]

    img_paths = [case["file"] + ".jpg" for case in cases]
    mask_paths = [case["file"] + ".tif" for case in cases]

    dataset = ResidueDataset(img_paths, mask_paths)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 12)
    )

    # Error colours:
    # 0 = correct
    # 1 = false positive
    # 2 = false negative
    error_cmap = ListedColormap([
        "black",
        "red",
        "deepskyblue"
    ])

    with torch.no_grad():
        for row, case in enumerate(cases):
            image, mask = dataset[row]

            output = model(image.unsqueeze(0).to(device))
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float()

            # Remove batch/channel dimensions for plotting
            image_display = image.permute(1, 2, 0).cpu()
            mask_display = mask.squeeze().cpu()
            pred_display = prediction.squeeze().cpu()

            intersection = (
                pred_display * mask_display
            ).sum()

            union = (
                (pred_display + mask_display) > 0
            ).float().sum()

            iou = (
                (intersection + 1e-7) /
                (union + 1e-7)
            ).item()

            true_coverage = mask_display.mean().item()
            pred_coverage = pred_display.mean().item()
            coverage_error = abs(
                pred_coverage - true_coverage
            )

            # Build error map
            error_map = torch.zeros_like(mask_display)

            false_positive = (
                (pred_display == 1) &
                (mask_display == 0)
            )

            false_negative = (
                (pred_display == 0) &
                (mask_display == 1)
            )

            error_map[false_positive] = 1
            error_map[false_negative] = 2

            # Original RGB image
            axes[row, 0].imshow(image_display)
            axes[row, 0].set_ylabel(
                case["name"],
                fontsize=12,
                fontweight="bold"
            )

            # Ground-truth mask
            axes[row, 1].imshow(
                mask_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            # Predicted mask
            axes[row, 2].imshow(
                pred_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            # False-positive / false-negative map
            axes[row, 3].imshow(
                error_map,
                cmap=error_cmap,
                vmin=0,
                vmax=2
            )

            axes[row, 0].set_title(
                f"{case['file'].split('/')[-1]}\n"
                f"IoU = {iou:.4f}"
            )

            axes[row, 1].set_title(
                f"Ground truth\n"
                f"Coverage = {true_coverage:.2%}"
            )

            axes[row, 2].set_title(
                f"Prediction\n"
                f"Coverage = {pred_coverage:.2%}"
            )

            axes[row, 3].set_title(
                f"Errors\n"
                f"Absolute error = {coverage_error:.2%}"
            )

            print(
                f"{case['name']}: "
                f"IoU={iou:.4f}, "
                f"true={true_coverage:.4f}, "
                f"pred={pred_coverage:.4f}, "
                f"error={coverage_error:.4f}"
            )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        "Representative Results on Test Location E\n"
        "Error map: red = false positive, blue = false negative",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    visualise_e()
