import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from json.decoder import JSONDecodeError

import numpy as np
import requests
import torch
from cfg import augmentations, device
from datasets import load_dataset
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer
from pycocotools import mask as mask_utils  


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_split="train",
        get_features=False,
        cache_path="/globalscratch/ucl/irec/darimez/dino",
        transform=augmentations,
        caching=False,
        segmentation=False,
        caption_sampling="first",
    ):
        super().__init__()
        self.caption_sampling = caption_sampling
        self.get_features = get_features

        self.cache_path = cache_path
        if caching:
            # self.dataset = load_dataset(
            #     "ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split=data_split
            # )
            cache_features(self)

        self.data_path = os.path.join(
            self.cache_path,
            "coco/images" if data_split == "train" else "coco/validation/images",
        )
        with open(
            "/globalscratch/ucl/irec/darimez/dino/coco/validation/annotations/instances_val2017.json",
            "r",
        ) as f:
            self.data = json.load(f)

        self.images = {img["id"]: img for img in self.data["images"]}
        self.annotations = self.data["annotations"]

        # map COCO category_id → YOLO index (0-based)
        self.id_to_yolo = {
            cat["id"]: idx for idx, cat in enumerate(self.data["categories"])
        }

        # group annotations by image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        # map id → filename
        self.id_to_file = {
            img_id: img_dict["file_name"].split(".")[0]
            for img_id, img_dict in self.images.items()
        }

        self.ids = list(self.images.keys())
        self.transform = transform
        self.clip = self.clip_tokenizer = None
        self.patch_size = 16
        self.seg = segmentation

    def __len__(self):
        return len(self.ids)

    @torch.no_grad()
    def __getitem__(self, idx):
        image = None

        image_id = self.ids[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.data_path, f"{self.id_to_file[image_id]}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        anns = self.img_to_anns.get(image_id, [])
        w, h = img_info["width"], img_info["height"]

        orig_w, orig_h = w, h
        out_h, out_w = image.shape[1], image.shape[2]

        labels = {"cls": [], "bboxes": []}
        if self.seg: 
            labels["masks"] = [] 

        for ann in anns:
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # [x_min, y_min, width, height]

            # normalize to YOLO format
            x_min, y_min, bw, bh = bbox
            x_center = (x_min + bw / 2) / w
            y_center = (y_min + bh / 2) / h
            bw /= w
            bh /= h

            yolo_class = self.id_to_yolo[cat_id]
            labels["cls"].append([yolo_class])
            labels["bboxes"].append([x_center, y_center, bw, bh])

            if self.seg: 
                seg = ann.get("segmentation", None)
                if seg is None:
                    # empty mask
                    mask_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
                else:
                    # polygon format (list of lists) or RLE
                    if isinstance(seg, list):
                        # rasterize polygons using PIL
                        mask_img = Image.new("L", (orig_w, orig_h), 0)
                        draw = ImageDraw.Draw(mask_img)
                        for poly in seg:
                            # poly is [x1,y1,x2,y2,...]
                            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                            draw.polygon(xy, outline=1, fill=1)
                        mask_arr = np.array(mask_img, dtype=np.uint8)
                    else:
                        # RLE (pycocotools)
                        rle = mask_utils.frPyObjects(seg, orig_h, orig_w)
                        mask_arr = mask_utils.decode(rle)
                        # decode may return (H,W) or (H,W,1); ensure 2D
                        if mask_arr.ndim == 3:
                            mask_arr = np.any(mask_arr, axis=2).astype(np.uint8)
                # convert to PIL to resize to transformed image size (nearest)
                mask_pil = Image.fromarray(mask_arr * 255).convert("L")
                if (out_w, out_h) != (orig_w, orig_h):
                    mask_pil = mask_pil.resize((out_w, out_h), resample=Image.NEAREST)
                mask_resized = np.array(mask_pil, dtype=np.uint8)
                # convert to 0/1
                mask_resized = (mask_resized > 127).astype(np.uint8)
                labels["masks"].append(torch.from_numpy(mask_resized))  # uint8 tensor HxW


        # convert to tensor
        labels["cls"] = torch.tensor(labels["cls"], dtype=torch.long) 
        labels["bboxes"] = torch.tensor(labels["bboxes"], dtype=torch.float)

        if self.seg:
            if len(labels["masks"]):
                # stack into (N, H, W) uint8 tensor
                labels["masks"] = torch.stack(labels["masks"]).to(dtype=torch.uint8)
            else:
                # no instances: empty tensor with correct spatial dims
                labels["masks"] = torch.empty((0, out_h, out_w), dtype=torch.uint8)

            if self.get_features:
                if self.caption_sampling == "first":
                    features = [
                        torch.load(f"{self.cache_path}/coco/captions/{id}_0.pt").detach()
                    ]
                elif self.caption_sampling == "random":
                    caption = np.random.choice(list(range(5)))
                    features = [
                        torch.load(
                            f"{self.cache_path}/coco/captions/{id}_{caption}.pt"
                        ).detach()
                    ]
                elif self.caption_sampling == "mean":
                    features = []
                    for caption in range(5):
                        features += [
                            torch.load(
                                f"{self.cache_path}/coco/captions/{id}_{caption}.pt"
                            ).detach()
                        ]

                return id, image, features

        return int(image_id), image, labels 

    
    def collate_fn(self, batch):
        if self.get_features:
            images = torch.stack([item[1] for item in batch])
            ids = [item[0] for item in batch]
            features = [item[2] for item in batch]
            return ids, images, features
        else:
            ids = [item[0] for item in batch]
            images = torch.stack([item[1] for item in batch], dim=0)

            labels = {"cls": [], "bboxes": [], "batch_idx": []}
            if "masks" in batch[0][2]:
                labels["masks"] = []      

            for batch_id, item in enumerate(batch):
                lab = item[2]
                n = len(lab["cls"])

                labels["cls"].append(lab["cls"])
                labels["bboxes"].append(lab["bboxes"])
                labels["batch_idx"].append(torch.full((n,), batch_id))

                if "masks" in lab:
                    labels["masks"].append(lab["masks"])

            labels["cls"] = torch.cat(labels["cls"])
            labels["bboxes"] = torch.cat(labels["bboxes"])
            labels["batch_idx"] = torch.cat(labels["batch_idx"])
            if labels["masks"]: 
                # masks_list is list of (n_i, H, W); concatenate along instance dim -> (N_total, H, W) 
                labels["masks"] = torch.cat(labels["masks"], dim=0) 
            else: 
                # no masks in batch 
                _, H, W = images.shape[0], images.shape[2], images.shape[3] 
                # not used; safer to infer from images 
                labels["masks"] = torch.empty((0, images.shape[2], images.shape[3]), dtype=torch.uint8) 

            return ids, images, labels 


if __name__ == "__main__":
    # Load CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = 16
    clip = CLIPModel.from_pretrained(
        f"openai/clip-vit-base-patch{patch_size}"
    ).text_model.to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        f"openai/clip-vit-base-patch{patch_size}"
    )

    dataset = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train")
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    def download_image(url, id, texts):
        save_path = f"/globalscratch/ucl/irec/darimez/dino/coco/images/{id}.png"
        save_caption_paths = [
            f"/globalscratch/ucl/irec/darimez/dino/coco/captions/{id}_{it}.pt"
            for it in range(len(texts))
        ]
        if not all(os.path.exists(path) for path in save_caption_paths + [save_path]):
            image = Image.open(BytesIO(requests.get(url).content))
            image.convert("RGB").save(save_path)
            for it, text in enumerate(texts):
                tokens = clip_tokenizer(text, return_tensors="pt")
                feature = clip(**tokens).pooler_output  #  last_hidden_state
                torch.save(feature, save_caption_paths[it])
        else:
            print(
                f"/!\\/!\\/!\\ Image and captions for {id} already exist /!\\/!\\/!\\"
            )

    def cache_features_old(dataset):
        tasks = {"id": [], "url": []}
        texts = {}
        for data in tqdm(dataset, total=len(dataset)):
            id = data["URL"].split("/")[-1].split(".")[0]
            save_path = f"/globalscratch/ucl/irec/darimez/dino/coco/images/{id}.png"
            save_caption_paths = [
                f"/globalscratch/ucl/irec/darimez/dino/coco/captions/{id}_{it}.pt"
                for it in range(len(texts))
            ]
            if not all(
                os.path.exists(path) for path in save_caption_paths + [save_path]
            ):
                url = data["URL"]
                if not id in tasks["id"]:
                    tasks["id"].append(id)
                    tasks["url"].append(url)
                if id in texts.keys():
                    texts[id].append(data["TEXT"])
                else:
                    texts[id] = [data["TEXT"]]

        # Download in parallel
        max_workers = 16  # adjust based on bandwidth and CPU
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_image, url, id, texts[id]): id
                for url, id in zip(tasks["url"], tasks["id"])
            }
            for f in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading images"
            ):
                _ = f.result()

    def cache_features(dataset):
        tasks = {"id": [], "url": [], "imgs": []}
        texts = {}
        dataset = json.loads(dataset)
        ids = sorted(
            os.listdir(os.path.join(self.cache_path, "coco/validation/images"))
        )
        for id in tqdm(ids, total=len(dataset)):
            id = int(id.split(".")[0])
            save_path = f"/globalscratch/ucl/irec/darimez/dino/coco/images/{id}.png"
            texts[id] = [
                data["caption"] for data in dataset if int(data["image_id"]) == id
            ]
            save_caption_paths = [
                f"/globalscratch/ucl/irec/darimez/dino/coco/validation/clip_captions/{id}_{it}.pt"
                for it in range(len(texts[id]))
            ]
            tasks["id"].append(id)
            tasks["img"].append(save_path)
            tasks["url"].append(save_caption_paths)

        # Download in parallel
        max_workers = 16  # adjust based on bandwidth and CPU
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_image_new, url, id, img, texts[id]): id
                for url, id, img in zip(tasks["url"], tasks["id"], tasks["imgs"])
            }
            for f in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading images"
            ):
                _ = f.result()

    cache_features(dataset)
