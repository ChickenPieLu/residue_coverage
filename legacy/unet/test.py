import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from legacy.unet import main, utils
from legacy.unet.dataset import ResidueDataset
from legacy.unet.evaluate import evaluate
from legacy.unet.model import MiniUNet

def test():
    # use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = MiniUNet().to(device)
    model.load_state_dict(
        torch.load(
            utils.DEFAULT_CHECKPOINT,
            map_location=device
        )
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    # evaluate model on testing set
    val_file_names = utils.read_file_names(['E'])
    val_img_paths = [p + ".jpg" for p in val_file_names]
    val_mask_paths = [p + ".tif" for p in val_file_names]

    val_dataset = ResidueDataset(val_img_paths,val_mask_paths)
    val_loader = DataLoader(
        val_dataset,
        batch_size= 4,
        shuffle= False,
        num_workers= 0
    )

    def combined_loss(logits, targets):
        bce = bce_criterion(logits, targets)
        dice = main.dice_loss(logits, targets)

        return bce + dice

    model.eval()

    print("testing set E:")
    evaluate(model, val_loader, combined_loss, device)

    all_true_coverage = []
    all_pred_coverage = []
    all_absolute_errors = []
    all_iou = []

    with torch.no_grad():
        image_index = 0

        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()

            true_coverage = masks.mean(dim=(1, 2, 3))
            pred_coverage = preds.mean(dim=(1, 2, 3))
            absolute_errors = torch.abs(pred_coverage - true_coverage)

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
            image_iou = (intersection + 1e-7) / (union + 1e-7)

            all_true_coverage.append(true_coverage.cpu())
            all_pred_coverage.append(pred_coverage.cpu())
            all_absolute_errors.append(absolute_errors.cpu())
            all_iou.append(image_iou.cpu())

            for i in range(len(imgs)):
                file_name = val_file_names[image_index]

                print(
                    f"{file_name}: "
                    f"IoU={image_iou[i].item():.4f}, "
                    f"true={true_coverage[i].item():.4f}, "
                    f"pred={pred_coverage[i].item():.4f}, "
                    f"error={absolute_errors[i].item():.4f}"
                )

                image_index += 1

    all_true_coverage = torch.cat(all_true_coverage)
    all_pred_coverage = torch.cat(all_pred_coverage)
    all_absolute_errors = torch.cat(all_absolute_errors)
    all_iou = torch.cat(all_iou)

    coverage_errors = all_pred_coverage - all_true_coverage

    coverage_mae = all_absolute_errors.mean()
    coverage_rmse = torch.sqrt((coverage_errors ** 2).mean())
    coverage_bias = coverage_errors.mean()

    print("\nPer-image results on E:")
    print(f"Mean IoU: {all_iou.mean():.4f}")
    print(f"IoU standard deviation: {all_iou.std():.4f}")
    print(f"Coverage MAE: {coverage_mae:.4f}")
    print(f"Coverage RMSE: {coverage_rmse:.4f}")
    print(f"Coverage bias: {coverage_bias:+.4f}")
    print(f"Maximum coverage error: {all_absolute_errors.max():.4f}")

if __name__ == "__main__":
    test()
