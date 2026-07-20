import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import argparse
from dataset import ResidueDataset
from model import MiniUNet
import subprocess

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_intersection = 0
    total_union = 0
    total_predicted = 0
    total_true = 0
    total_pixels = 0
    max_probability = 0

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
            total_union += ((preds + masks) > 0).float().sum().item()

            total_predicted += preds.sum().item()
            total_true += masks.sum().item()
            total_pixels += masks.numel()

            max_probability = max(
                max_probability,
                probabilities.max().item()
            )

    average_loss = total_loss / len(loader.dataset)
    iou = total_intersection / (total_union + 1e-8)
    pred_ratio = total_predicted / total_pixels
    true_ratio = total_true / total_pixels

    print(
        f"loss={average_loss:.4f}, "
        f"IoU={iou:.4f}, "
        f"true_ratio={true_ratio:.4f}, "
        f"pred_ratio={pred_ratio:.4f}, "
        f"max_prob={max_probability:.4f}"
    )


def main(args):
    # use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    training_dirs = ['A','B','C']
    file_names = utils.read_file_names(training_dirs)

    img_paths = [p + ".jpg" for p in file_names]
    mask_paths = [p + ".tif" for p in file_names]

    # if training (default)
    if not args.val:
        dataset = ResidueDataset(img_paths, mask_paths)
        print(f"Number of training pairs: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )

        model = MiniUNet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimiser = optim.Adam(
            model.parameters(),
            lr = 0.001
        )

        for epoch in range(20):
            print(f"epoch: {epoch}")

            model.train()
            total_loss = 0

            for batch_index, (imgs, masks) in enumerate(loader):
                imgs = imgs.to(device)
                masks = masks.to(device)

                optimiser.zero_grad()

                output = model(imgs)
                loss = criterion(output,masks)

                loss.backward()
                optimiser.step()

                # (average loss of 1 img in batch) * (# of imgs in a batch)
                total_loss += loss.item() * imgs.size(0) 

                if batch_index % 20 == 0:
                    print(f"batch {batch_index}/{len(loader)}")
            
            # avg loss of the full training set
            avg_loss = total_loss/len(dataset)
            print(f"Average loss: {avg_loss:.6f}\n")

        print("Training set:")
        evaluate(model, loader, criterion, device)
        torch.save(
            model.state_dict(),
            "mini_unet_abc.pth"
        )

    # no training: read stored model
    else:
        model = MiniUNet().to(device)
        model.load_state_dict(
            torch.load(
                "mini_unet_abc.pth",
                map_location=device
            )
        )
        criterion = nn.BCEWithLogitsLoss()
        optimiser = optim.Adam(
            model.parameters(),
            lr = 0.001
        )

    # evaluate model on validation set
    val_file_names = utils.read_file_names(['D'])
    val_img_paths = [p + ".jpg" for p in val_file_names]
    val_mask_paths = [p + ".tif" for p in val_file_names]

    val_dataset = ResidueDataset(val_img_paths,val_mask_paths)
    val_loader = DataLoader(
        val_dataset,
        batch_size= 4,
        shuffle= False,
        num_workers= 0
    )

    print("Validation set D:")
    evaluate(model, val_loader, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--val",action="store_true",default=False)
    args = parser.parse_args()

    main(args)

    subprocess.run([
        "osascript",
        "-e",
        'display notification "Model training completed" with title "Python"'
    ])