import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import main
import utils
from dataset import ResidueDataset
from model import MiniUNet

def validate():
    # use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = MiniUNet().to(device)
    model.load_state_dict(
        torch.load(
            "mini_unet_abc.pth",
            map_location=device
        )
    )
    criterion = nn.BCEWithLogitsLoss()

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
    main.evaluate(model, val_loader, criterion, device)
    with torch.no_grad():
        for batch_index, (imgs, masks) in enumerate(val_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)

            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()

            # true_coverage = masks.mean(dim=(1, 2, 3))
            # pred_coverage = preds.mean(dim=(1, 2, 3))

            # absolute_errors = torch.abs(
            #     pred_coverage - true_coverage
            # )
            # print(f"batch {batch_index}:")
            # print(f"MAE: {torch.mean(absolute_errors):.6f}")
            if batch_index == 17:
                utils.visualise(imgs[0],masks[0],preds[0])

if __name__ == "__main__":
    validate()