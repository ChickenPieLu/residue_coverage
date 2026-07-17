import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import argparse
from dataset import ResidueDataset
from model import MiniUNet

def main(args):
    file_names = utils.read_file_names(args.dir)

    img_paths = [p + ".jpg" for p in file_names]
    mask_paths = [p + ".tif" for p in file_names]

    dataset = ResidueDataset(img_paths, mask_paths)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    imgs,masks = next(iter(loader))

    model = MiniUNet()
    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(
        model.parameters(),
        lr = 0.001
    )

    model.train()

    for epoch in range(51):
        optimiser.zero_grad()

        output = model(imgs)
        loss = criterion(output,masks)

        loss.backward()
        optimiser.step()

        if epoch % 10 == 0:
            print(f"epoch: {epoch}")
            print(f"loss: {loss:.6f}")

    model.eval()

    with torch.no_grad():
        outputs = model(imgs)
        probability = torch.sigmoid(outputs)
        preds = (probability>0.5).float()

    intersection = (preds * masks).sum()
    union = ((preds + masks) > 0).float().sum()
    iou = intersection / (union + 1e-8)

    print(
        f"loss={loss.item():.4f}, "
        f"IoU={iou.item():.4f}, "
        f"true_ratio={masks.mean().item():.4f}, "
        f"pred_ratio={preds.mean().item():.4f}, "
        f"max_prob={probability.max().item():.4f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",type=str,default="a")
    args = parser.parse_args()

    d = args.dir.upper()
    if not d in ['A','B','C','D']:
        raise ValueError(f"{d} must be one of A,B,C,D")
    args.dir = d
    main(args)