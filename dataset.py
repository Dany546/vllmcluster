from pydoc import text
from datasets import load_dataset
import torch, os
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPModel
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm
import numpy as np
from cfg import device, augmentations
from concurrent.futures import ThreadPoolExecutor, as_completed


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, data_split="train", transform=augmentations, caching=False):
        super().__init__()
        self.cache_path = "/globalscratch/ucl/irec/darimez/dino"
        self.dataset = load_dataset(
            "ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split=data_split
        )
        if caching:
            cache_features(self)
        self.transform = transform
        self.clip = self.clip_tokenizer = None
        self.patch_size = 16

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = None
        data = self.dataset[idx]
        caption = data["TEXT"]
        url = data["URL"]
        id = url.split("/")[-1].split(".")[0]
        if os.path.exists(f"{self.cache_path}/coco/captions/{id}.pt"):
            features = torch.load(f"{self.cache_path}/coco/captions/{id}.pt").detach()
        else:
            if self.clip is None:
                self.clip = CLIPModel.from_pretrained(
                    f"openai/clip-vit-base-patch{self.patch_size}"
                ).text_model
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                    f"openai/clip-vit-base-patch{self.patch_size}"
                )
            with torch.no_grad():
                tokens = self.clip_tokenizer(caption, return_tensors="pt")
                features = self.clip(**tokens).pooler_output  #  last_hidden_state
            torch.save(features, f"{self.cache_path}/coco/captions/{id}.pt")

        if os.path.exists(f"{self.cache_path}/coco/images/{id}.png"):
            image = Image.open(f"{self.cache_path}/coco/images/{id}.png")
        else:
            image = Image.open(BytesIO(requests.get(url).content))
            image = image.convert("RGB")
            image.save(f"{self.cache_path}/coco/images/{id}.png")

        if self.transform:
            image = self.transform(image)

        return image, features


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

    dataset = COCODataset(data_split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    def download_image(url, id, texts):
        save_path = f"{dataset.cache_path}/coco/images/{id}.png"
        image = Image.open(BytesIO(requests.get(url).content))
        image = image.convert("RGB")
        image.save(save_path)
        for it, text in enumerate(texts):
            tokens = clip_tokenizer(text, return_tensors="pt")
            feature = clip(**tokens).pooler_output  #  last_hidden_state
            torch.save(feature, f"{dataset.cache_path}/coco/captions/{id}_{it}.pt")

    def cache_features(dataset):
        tasks = {"id": [], "url": []}
        texts = {}
        for data in tqdm(dataset.dataset, total=len(dataset.dataset)):
            id = data["URL"].split("/")[-1].split(".")[0]
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

    cache_features(dataset)
