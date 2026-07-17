import torch
import torch.nn as nn
import torch.optim as optim
import utils
import argparse

def main(args):
    file_names = utils.read_file_names(args.dir)
    train_pairs = [
        [utils.jpg_read(name+".jpg"),utils.tif_read(name+".tif")] for name in file_names
    ]
    img = train_pairs[0][0]
    mask = train_pairs[0][1]
    print(img.shape)
    print(img.dtype)
    print(img.min(),img.max())
    print()

    print(mask.shape)
    print(mask.dtype)
    print(mask.unique())
    utils.visualise(img,mask)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",type=str,default="a")
    args = parser.parse_args()

    d = args.dir.upper()
    if not d in ['A','B','C','D']:
        raise ValueError(f"{d} must be one of A,B,C,D")
    
    main(args)