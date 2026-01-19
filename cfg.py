import os
from logging import logMultiprocessing

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import PngImagePlugin
from torchvision import transforms




PngImagePlugin.MAX_TEXT_CHUNK = 2 * 1024 * 1024  # must be increased to avoid errors
device = "cuda" if torch.cuda.is_available() else "cpu"

# os.environ["WANDB_MODE"] = "offline"  # ensures offline logging
model_cache_path = "/globalscratch/ucl/irec/darimez/dino/models/"
model_ckpt_path = model_cache_path + "dino2clip_COCO/"
if not os.path.exists(model_ckpt_path):
    os.makedirs(model_ckpt_path)

# augmentations = transforms.Compose(
#     [
#         transforms.Resize((518, 518)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomAffine(
#             degrees=20,  # rotation ±15°
#             translate=(0.1, 0.1),  # up to ±10% shift horizontally & vertically
#             scale=(0.9, 1.1),  # zoom in/out by ±10%
#             shear=(-10, 10),  # shear by ±10°
#             interpolation=transforms.InterpolationMode.BILINEAR,
#             fill=0,  # background fill color (0=black)
#         ),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# augmentations_yolo = transforms.Compose(
#     [
#         transforms.Resize((640, 640)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomAffine(
#             degrees=20,  # rotation ±15°
#             translate=(0.1, 0.1),  # up to ±10% shift horizontally & vertically
#             scale=(0.9, 1.1),  # zoom in/out by ±10%
#             shear=(-10, 10),  # shear by ±10°
#             interpolation=transforms.InterpolationMode.BILINEAR,
#             fill=0,  # background fill color (0=black)
#         ),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
#         transforms.ToTensor(),
#     ]
# )

import albumentations as A
from albumentations.pytorch import ToTensorV2

def make_albumentations_pipeline(size, normalize=False, convert_to_tensor=True):
    """
    Create an Albumentations pipeline for image preprocessing.
    
    Args:
        size: Target size for longest edge
        normalize: Whether to apply ImageNet normalization (mean/std subtraction)
        convert_to_tensor: If True, converts to CHW tensor. If False, keeps as HWC numpy array (for YOLO)
    """
    transforms = [
        A.LongestMaxSize(max_size=size), 
        A.PadIfNeeded(min_height=size, min_width=size, 
                      position="center", border_mode=0, 
                      fill=114, fill_mask=0), 
    ]
    # optional photometric augmentations (add as needed)
    # transforms += [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)]

    if normalize:
        transforms += [A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))]
    
    # Convert to tensor [0, 1] float32 CHW (unless skipped for YOLO)
    # YOLO needs HWC uint8 to use its own internal processing
    # DINO/CLIP need CHW float32 tensors
    if convert_to_tensor:
        transforms += [ToTensorV2()]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_area=1, min_visibility=0.0),
        additional_targets={"masks": "masks"}  # masks passed as list of HxW arrays
    )

augmentations = {
    "yolo": make_albumentations_pipeline(640, normalize=False, convert_to_tensor=False),  # YOLO: HWC uint8, handles its own preprocessing
    "dino": make_albumentations_pipeline(518, normalize=True, convert_to_tensor=True),     # DINO: CHW [0,1] normalized
    "clip": make_albumentations_pipeline(224, normalize=False, convert_to_tensor=True),    # CLIP: CHW [0,1]
}


# YOLO index -> COCO supercategory
supercat_names = [
    "person",
    "vehicle",
    "outdoor",
    "animal",
    "accessory",
    "sports",
    "kitchen",
    "food",
    "furniture",
    "electronic",
    "indoor",
    "indoor",
]
cat_to_super = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 3,
    22: 3,
    23: 3,
    24: 4,
    25: 4,
    26: 4,
    27: 4,
    28: 4,
    29: 5,
    30: 5,
    31: 5,
    32: 5,
    33: 5,
    34: 5,
    35: 5,
    36: 5,
    37: 5,
    38: 5,
    39: 6,
    40: 6,
    41: 6,
    42: 6,
    43: 6,
    44: 6,
    45: 6,
    46: 7,
    47: 7,
    48: 7,
    49: 7,
    50: 7,
    51: 7,
    52: 7,
    53: 7,
    54: 7,
    55: 7,
    56: 8,
    57: 8,
    58: 8,
    59: 8,
    60: 8,
    61: 8,
    62: 9,
    63: 9,
    64: 9,
    65: 9,
    66: 9,
    67: 9,
    68: 10,
    69: 10,
    70: 10,
    71: 10,
    72: 10,
    73: 11,
    74: 11,
    75: 11,
    76: 11,
    77: 11,
    78: 11,
    79: 11,
}
