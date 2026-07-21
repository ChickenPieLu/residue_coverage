import torch

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_intersection = 0
    total_union = 0
    total_predicted = 0
    total_true = 0
    total_pixels = 0
    max_probability = 0
    all_absolute_errors = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()

            total_loss += loss.item() * imgs.size(0)
            total_intersection += (preds * masks).sum().item()
            total_union += (
                (preds + masks) > 0
            ).float().sum().item()

            total_predicted += preds.sum().item()
            total_true += masks.sum().item()
            total_pixels += masks.numel()

            max_probability = max(
                max_probability,
                probabilities.max().item()
            )

            true_coverage = masks.mean(dim=(1, 2, 3))
            pred_coverage = preds.mean(dim=(1, 2, 3))

            absolute_errors = torch.abs(
                pred_coverage - true_coverage
            )

            all_absolute_errors.extend(
                absolute_errors.cpu().tolist()
            )

    average_loss = total_loss / len(loader.dataset)
    iou = total_intersection / (total_union + 1e-8)

    precision = total_intersection / (
        total_predicted + 1e-8
    )
    recall = total_intersection / (
        total_true + 1e-8
    )
    dice = 2 * total_intersection / (
        total_predicted + total_true + 1e-8
    )

    pred_ratio = total_predicted / total_pixels
    true_ratio = total_true / total_pixels

    coverage_mae = (
        sum(all_absolute_errors)
        / len(all_absolute_errors)
    )

    print(
        f"loss={average_loss:.4f}, "
        f"IoU={iou:.4f}, "
        f"Dice={dice:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"Coverage_MAE={coverage_mae:.4f}, "
        f"true_ratio={true_ratio:.4f}, "
        f"pred_ratio={pred_ratio:.4f}, "
        f"max_prob={max_probability:.4f}"
    )

    return iou