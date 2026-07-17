from torch.utils.data import Dataset
import utils

class ResidueDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        if len(img_paths) != len(mask_paths):
            raise ValueError(
                f"Images and masks have different lengths: "
                f"{len(img_paths)} vs {len(mask_paths)}"
            )

        self.img_paths = img_paths
        self.mask_paths = mask_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        img = utils.jpg_read(img_path)
        mask = utils.tif_read(mask_path)

        return img, mask