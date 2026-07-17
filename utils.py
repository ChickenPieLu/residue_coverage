import os
import cv2
import torch
import tifffile
import matplotlib as plt

def read_file_names(directory):
    full_path = "residue_background/"+directory
    file_names = [
        os.path.join(full_path,f[:-4]) for f in os.listdir(full_path)
        if f.lower().endswith(".jpg")
    ]
    return file_names

def jpg_read(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Can't read img: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = torch.from_numpy(img_rgb)
    image = image.permute(2,0,1) #(h,w,3) to (3,h,w)
    image = image.float()/255.0
    return image 

def tif_read(path):
    mask = tifffile.imread(path)
    if mask is None:
        raise ValueError(f"Can't read mask: {path}")
    
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0) #(h,w) to (1,h,w)
    mask = mask.float()/255.0
    return mask

def visualise(image,mask):
    import matplotlib.pyplot as plt

    image_to_show = (
        image
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    mask_to_show = (
        mask
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_to_show)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_to_show, cmap="gray", vmin=0, vmax=1)
    plt.title("Mask")
    plt.axis("off")

    plt.show()