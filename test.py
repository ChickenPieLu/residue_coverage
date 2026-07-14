import utils
import argparse
import numpy as np
from pathlib import Path

def main(args,dirs):
    img_paths, mask_paths = utils.read_paths(dirs)
    img = utils.jpg_read(img_paths[args.num])
    mask = utils.tiff_read(mask_paths[args.num])
    utils.show_plt(np.clip(img + np.stack([mask*100,np.zeros_like(mask),np.zeros_like(mask)],axis=-1),0,255))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num",type=int,default=0)
    parser.add_argument("--dir",type=str,default="a")
    args = parser.parse_args()
    d = args.dir.upper()
    if (d not in ['A','B','C','D',"E"]) or len(d)!=1:
        raise ValueError(f"{d} 应该是A,B,C,D,E其中之一")
    dirs = ["residue_background/"+d,]
    main(args,dirs)