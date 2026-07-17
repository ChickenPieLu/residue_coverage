import torch
import torch.nn as nn
import torch.optim as optim
import utils
import argparse
from dataset import ResidueDataset

def main(args):
    file_names = utils.read_file_names(args.dir)

    img_paths = [p + ".jpg" for p in file_names]
    mask_paths = [p + ".tif" for p in file_names]

    dataset = ResidueDataset(img_paths, mask_paths)

    img, mask = dataset[0]

    print("Dataset length:", len(dataset))
    print("Image shape:", img.shape)
    print("Image dtype:", img.dtype)
    print("Image range:", img.min(), img.max())

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", mask.unique())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",type=str,default="a")
    args = parser.parse_args()

    d = args.dir.upper()
    if not d in ['A','B','C','D']:
        raise ValueError(f"{d} must be one of A,B,C,D")
    args.dir = d
    main(args)