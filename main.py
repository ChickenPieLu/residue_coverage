import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import utils
from dataset import ResidueDataset
from model import MiniUNet
import subprocess
from evaluate import evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # random seed to fix random process
    SEED = 114514
    set_seed(SEED)
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    # use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)


    # training set loader
    training_dirs = ['A','B','C']
    file_names = utils.read_file_names(training_dirs)

    img_paths = [p + ".jpg" for p in file_names]
    mask_paths = [p + ".tif" for p in file_names]

    dataset = ResidueDataset(img_paths, mask_paths)
    print(f"Number of training pairs: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=train_generator
    )

    # validation set loader
    val_file_names = utils.read_file_names(['D'])
    val_img_paths = [p + ".jpg" for p in val_file_names]
    val_mask_paths = [p + ".tif" for p in val_file_names]

    val_dataset = ResidueDataset(val_img_paths,val_mask_paths)
    val_loader = DataLoader(
        val_dataset,
        batch_size= 4,
        shuffle= False,
        num_workers= 0,
    )

    # model
    model = MiniUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(
        model.parameters(),
        lr = 0.001
    )

    #training
    best_val_iou = -1.0
    best_epoch = -1

    patience = 10
    epochs_without_improvement = 0

    checkpoint_path = (
        "mini_unet_abc_bce_seed114514_train_generator.pth"
    )

    for epoch in range(50):
        print(f"\nEpoch {epoch}")

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
        
        # avg loss of the full training set
        avg_loss = total_loss/len(dataset)
        print(f"Average loss: {avg_loss:.6f}")
        print("Validation on D:")
        val_iou = evaluate(model,val_loader,criterion,device)

        # patience and early stop due to non-improvment
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"(best IoU so far: {best_val_iou:.4f}, "
                f"epoch {best_epoch})"
            )

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{patience} epochs"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping. Best epoch: {best_epoch}, "
                f"best IoU: {best_val_iou:.4f}"
            )
            break

    # final evaluation
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    print("\nBest model on training set:")
    evaluate(model, loader, criterion, device)

    print("\nBest model on validation set D:")
    evaluate(model, val_loader, criterion, device)

if __name__ == "__main__":
    main()

    subprocess.run([
        "osascript",
        "-e",
        'display notification "main.py execution completed" with title "Python"'
    ])