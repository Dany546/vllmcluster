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
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_split="train",
        get_features=False,
        cache_path="/globalscratch/ucl/irec/darimez/dino",
        transform=augmentations,
        caching=False,
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

        labels = {"cls": [], "bboxes": []}
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

        # convert to tensor
        for label in labels.keys():
            labels[label] = torch.from_numpy(np.array(labels[label]))

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
            labels = {}
            labels["cls"] = torch.cat([item[2]["cls"] for item in batch])
            labels["bboxes"] = torch.cat([item[2]["bboxes"] for item in batch])
            batch_ids = []
            for batch_id, item in enumerate(batch):
                batch_ids += [torch.ones(len(item[2]["cls"])) * batch_id]
            labels["batch_idx"] = torch.cat(batch_ids, dim=0)
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
