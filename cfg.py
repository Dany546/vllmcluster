import torch, os
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["WANDB_MODE"] = "offline"  # ensures offline logging
model_cache_path = "/globalscratch/ucl/irec/darimez/dino/models/"
model_ckpt_path = model_cache_path + "dino2clip_COCO/"

augmentations = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=20,  # rotation ±15°
            translate=(0.1, 0.1),  # up to ±10% shift horizontally & vertically
            scale=(0.9, 1.1),  # zoom in/out by ±10%
            shear=(-10, 10),  # shear by ±10°
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,  # background fill color (0=black)
        ),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
