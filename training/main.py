import argparse
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training import utils
from training.dataset import ResidueDataset
from model import MODEL_NAME, make_model
from training.evaluate import evaluate


DEFAULT_RETRAINED_CHECKPOINT = (
    utils.PROJECT_ROOT
    / "smp_unet_resnet34_training_candidate_seed42.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SMP U-Net using A/B/C and validate on D."
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=DEFAULT_RETRAINED_CHECKPOINT,
        help=(
            "Destination for the best checkpoint. Defaults to a new "
            "retrained model in the project root."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Show a macOS notification after training finishes.",
    )
    return parser.parse_args()


def choose_device(requested):
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


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    dimensions = tuple(range(1, probabilities.ndim))

    intersection = (
        probabilities * targets
    ).sum(dim=dimensions)

    total = (
        probabilities.sum(dim=dimensions)
        + targets.sum(dim=dimensions)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        total + smooth
    )

    return 1.0 - dice.mean()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    # random seed to fix random process
    SEED = 42
    set_seed(SEED)
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    device = choose_device(args.device)
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
        batch_size=args.batch_size,
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
        batch_size=args.batch_size,
        shuffle= False,
        num_workers= 0,
    )

    # model
    model = make_model().to(device)
    print("Model:", MODEL_NAME)
    bce_criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    def combined_loss(logits, targets):
        bce = bce_criterion(logits, targets)
        dice = dice_loss(logits, targets)

        return bce + dice

    #training
    best_val_iou = -1.0
    best_epoch = -1

    patience = args.patience
    epochs_without_improvement = 0

    checkpoint_path = args.output_checkpoint.expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path == utils.DEFAULT_CHECKPOINT.resolve():
        raise ValueError(
            "Refusing to overwrite the existing production checkpoint. "
            "Choose a different --output-checkpoint path."
        )
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing checkpoint: {checkpoint_path}"
        )
    print("Checkpoint:", checkpoint_path)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}")

        model.train()
        total_loss = 0
        total_bce = 0
        total_dice = 0

        for batch_index, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimiser.zero_grad()

            outputs = model(imgs)
            bce = bce_criterion(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce + dice

            loss.backward()
            optimiser.step()

            batch_size = imgs.size(0)

            total_loss += loss.item() * batch_size
            total_bce += bce.item() * batch_size
            total_dice += dice.item() * batch_size
        
        # loss print
        dataset_size = len(loader.dataset)
        print(
            f"Average loss: {total_loss / dataset_size:.6f}, "
            f"BCE: {total_bce / dataset_size:.6f}, "
            f"Dice loss: {total_dice / dataset_size:.6f}"
        )

        #evaluation print
        val_iou = evaluate(model,val_loader,combined_loss,device)

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
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    print("\nBest model on training set:")
    evaluate(model, loader, combined_loss, device)

    print("\nBest model on validation set D:")
    evaluate(model, val_loader, combined_loss, device)

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)

    if arguments.notify:
        subprocess.run([
            "osascript",
            "-e",
            (
                'display notification "training completed" '
                'with title "Residue Coverage"'
            ),
        ], check=False)
