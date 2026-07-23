from pathlib import Path

import tifffile
import cv2
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "residue_background"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "residue_rf_model.joblib"

# 路径读取
def read_paths(dirs):
    """Return image/mask paths paired by stem, independent of working directory."""
    img_paths, mask_paths = [], []
    for directory in dirs:
        directory = Path(directory)
        if len(directory.parts) == 1 and directory.name.upper() in "ABCDE":
            directory = DATA_ROOT / directory.name.upper()
        elif not directory.is_absolute():
            directory = PROJECT_ROOT / directory

        images = {path.stem: path for path in directory.glob("*.jpg")}
        masks = {path.stem: path for path in directory.glob("*.tif")}
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            raise ValueError(
                f"{directory} 中图片和 mask 无法按文件名配对；"
                f"缺少 mask: {missing_masks[:3]}，"
                f"缺少图片: {missing_images[:3]}"
            )

        for stem in sorted(images):
            img_paths.append(str(images[stem]))
            mask_paths.append(str(masks[stem]))

    return img_paths, mask_paths


def location_dirs(locations):
    return [DATA_ROOT / location.upper() for location in locations]


def display_name(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT).with_suffix(""))
    except ValueError:
        return str(path.with_suffix(""))

# 图片读取
def tiff_read(path):
    result = tifffile.imread(path)
    if result is None:
        raise ValueError(f"无法读取图片：{path}")
    
    return result > 0 # (h,w) 0 or 1

def jpg_read(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图片：{path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb # (h,w,c) 0-255

# 测试用数组
def test_rgb_generator(size = (20,20)):
    return np.stack(
        [np.random.randint(255,size=size),
        np.random.randint(255,size=size),
        np.random.randint(255,size=size)
        ],
        axis = -1
        ).astype(np.uint8)

def test_gray_generator(size = (20,20)):
    return np.random.randint(255, size=size).astype(np.uint8)

# 红色mask
def coloured_mask(mask):
    empty = np.full(mask.shape, 0).astype(np.uint8)
    coloured_mask = np.stack([mask*120,empty,empty],axis = -1)
    return coloured_mask # (h,w,c) 0-255

# 展示
def show_plt(img):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# 黑底白特征<=>白底黑特征
def tiff_converter():
    dir = ["residue_background/Zak-W-winterBarley_1m_20220401/IMG_0938",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0939",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0940",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0941",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0942",
           "residue_background/Zak-W-winterBarley_1m_20220401/IMG_0944",]
    _, mask_paths = read_paths(dir)
    for i in range(len(mask_paths)):
        img_path = mask_paths[i]
        mask = tifffile.imread(img_path)
        result = np.where(mask>0, 0, 255).astype(np.uint8)
        tifffile.imwrite(img_path,result,compression="lzw")

def time_convert(start, end):
    temp = end-start
    result = f"{int(temp//3600)}h "
    temp %= 3600
    result += f"{int(temp//60)}m "
    temp %= 60
    result += f"{temp:.2f}s"
    return result

def int_0_to_100(value: str) -> int:
    number = int(value)

    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError(
            f"{value} must be an integer between 0 and 100"
        )

    return number

def process_seq(seq):
    seq_list = [x.upper() for x in list(seq)]
    length = len(seq_list)
    allowed = ['A','B','C','D']
    if len(seq)<2:
        raise ValueError(seq+" 应多于2个字母")

    if len(list(set(seq_list))) != length:
        raise ValueError(seq+" shouldn't repeat any letter")
    
    train_dir = []
    for i in range(length-1):
        d = seq_list[i]
        if d not in allowed:
            raise ValueError(seq+" 只能包含A,B,C,D")
        train_dir.append(DATA_ROOT / d)
    
    test = seq_list[length-1]
    if test not in allowed:
            raise ValueError(seq+" 只能包含A,B,C,D")
    test_dir = [DATA_ROOT / test]
    
    return train_dir,test_dir
