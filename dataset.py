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
 
import albumentations as A
import torchvision.transforms as T

def _infer_pad_size_from_transform(transform, fallback=None):
    """
    Try to infer the final square pad size D from the provided transform.
    Supports:
      - Albumentations Compose with PadIfNeeded(min_height/min_width) or LongestMaxSize + PadIfNeeded
      - Custom PadToSquare transform with attribute `size`
      - torchvision-like transform where a PadToSquare instance is present
    Returns int or fallback (if provided) or raises ValueError.
    """
    # 1) Albumentations Compose
    try:
        # Albumentations Compose has attribute 'transforms' (list)
        if isinstance(transform, A.core.composition.Compose):
            # search for PadIfNeeded first
            for t in transform.transforms:
                # PadIfNeeded has class name 'PadIfNeeded'
                if getattr(t, "__class__", None).__name__ == "PadIfNeeded":
                    # PadIfNeeded stores min_height/min_width
                    mh = getattr(t, "min_height", None)
                    mw = getattr(t, "min_width", None)
                    if mh is not None and mw is not None and mh == mw:
                        return int(mh)
                    # if only one is set or they differ, prefer max
                    if mh is not None and mw is not None:
                        return int(max(mh, mw))
                # LongestMaxSize sets max_size; if followed by PadIfNeeded, we can use that PadIfNeeded
                if getattr(t, "__class__", None).__name__ == "LongestMaxSize":
                    max_size = getattr(t, "max_size", None)
                    # if max_size exists and there is a PadIfNeeded later, prefer PadIfNeeded; otherwise use max_size
                    # look ahead for PadIfNeeded
                    # (we already loop transforms in order; check subsequent transforms)
                    # find index
                    # safe guard: transforms is a list-like
                    try:
                        idx = transform.transforms.index(t)
                        for t2 in transform.transforms[idx + 1:]:
                            if getattr(t2, "__class__", None).__name__ == "PadIfNeeded":
                                mh = getattr(t2, "min_height", None)
                                mw = getattr(t2, "min_width", None)
                                if mh is not None and mw is not None and mh == mw:
                                    return int(mh)
                                if mh is not None and mw is not None:
                                    return int(max(mh, mw))
                        # no PadIfNeeded found, use max_size
                        if max_size is not None:
                            return int(max_size)
                    except Exception:
                        if max_size is not None:
                            return int(max_size)

    except Exception:
        # not albumentations or unexpected structure
        pass

    # 2) torchvision / custom PadToSquare
    # If transform is a torchvision.transforms.Compose, search its transforms list
    try:
        # torchvision Compose stores transforms in .transforms
        seq = getattr(transform, "transforms", None)
        if seq is not None:
            for t in seq:
                # check for attribute 'size' (your PadToSquare has .size)
                if hasattr(t, "size"):
                    try:
                        return int(getattr(t, "size"))
                    except Exception:
                        pass
                # also check class name
                if getattr(t, "__class__", None).__name__ == "PadToSquare":
                    size = getattr(t, "size", None)
                    if size is not None:
                        return int(size)
    except Exception:
        pass

    # 3) If transform itself has attribute 'size' (maybe user passed PadToSquare directly)
    if hasattr(transform, "size"):
        try:
            return int(getattr(transform, "size"))
        except Exception:
            pass

    # 4) fallback
    if fallback is not None:
        return int(fallback)

    raise ValueError(
        "Could not infer pad_size from transform. "
        "Please pass pad_size explicitly or use an Albumentations Compose with PadIfNeeded/LongestMaxSize "
        "or a PadToSquare transform with attribute 'size'."
    )


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
            img_id: img_dict["file_name"]
            for img_id, img_dict in self.images.items()
        }

        self.ids = list(self.images.keys()) 
        self.transform = transform 
        self.seg = segmentation
        self.caching = caching
        self.pad_size = _infer_pad_size_from_transform(transform, fallback=640)

    def __len__(self):
        return len(self.ids)

    def _ann_to_mask(self, ann, orig_w, orig_h):
        seg = ann.get("segmentation", None)
        if seg is None:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)
        if isinstance(seg, list):
            # polygon(s)
            mask_img = Image.new("L", (orig_w, orig_h), 0)
            draw = ImageDraw.Draw(mask_img)
            for poly in seg:
                xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                draw.polygon(xy, outline=1, fill=1)
            return np.array(mask_img, dtype=np.uint8)
        # RLE
        rle = mask_utils.frPyObjects(seg, orig_h, orig_w)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        return m.astype(np.uint8)

    @torch.no_grad()
    def __getitem__(self, idx):
        image_id = self.ids[idx] 
        img_path = os.path.join(self.data_path, self.id_to_file[image_id])
        orig_image = Image.open(img_path).convert("RGB")  
        orig_w, orig_h = orig_image.size

        anns = self.img_to_anns.get(image_id, [])

        # build COCO-format bboxes and category_ids
        bboxes = []
        category_ids = []
        masks = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # skip degenerate boxes
            if w <= 0 or h <= 0:
                continue
            bboxes.append([x, y, w, h])
            category_ids.append(self.id_to_yolo[ann["category_id"]])
            if self.seg:
                masks.append(self._ann_to_mask(ann, orig_w, orig_h))

        # Albumentations expects numpy HWC
        image_np = np.array(orig_image) 

        # If there are no boxes, pass empty lists (albumentations handles them)
        if self.seg:
            transformed = self.transform(image=image_np, bboxes=bboxes, masks=masks, category_ids=category_ids)
        else:
            transformed = self.transform(image=image_np, bboxes=bboxes, category_ids=category_ids)


        image = transformed["image"]
        # Convert numpy array to torch tensor if needed (for YOLO: HWC uint8 -> CHW float32 [0-1])
        # DINO/CLIP already converted by ToTensorV2 in augmentation pipeline
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0  # HWC uint8 -> CHW [0, 1]
        else:
            image = image.to(torch.float32)  # Already a tensor, just ensure float32
        image_t = image

        transformed_bboxes = transformed.get("bboxes", [])
        transformed_labels = transformed.get("category_ids", [])

        # convert bboxes (COCO pixel) -> YOLO normalized relative to pad_size
        D = self.pad_size
        yolo_bboxes = []
        for (x, y, w, h) in transformed_bboxes:
            x_center = (x + w / 2.0) / D
            y_center = (y + h / 2.0) / D
            bw = w / D
            bh = h / D
            yolo_bboxes.append([x_center, y_center, bw, bh])

        # tensors for cls and bboxes
        if len(yolo_bboxes):
            labels_cls = torch.tensor([[int(c)] for c in transformed_labels], dtype=torch.long)
            labels_bboxes = torch.tensor(yolo_bboxes, dtype=torch.float32)
        else:
            labels_cls = torch.empty((0,1), dtype=torch.long)
            labels_bboxes = torch.empty((0,4), dtype=torch.float32)

        labels = {"cls": labels_cls, "bboxes": labels_bboxes}

        if self.seg:
            masks = transformed.get("masks", [])
            if len(masks):
                # Convert masks to tensors (they may be numpy arrays or tensors depending on transform)
                mask_tensors = []
                for m in masks:
                    if isinstance(m, torch.Tensor):
                        mask_tensors.append(m.to(torch.uint8))
                    else:
                        # numpy array
                        mask_tensors.append(torch.from_numpy(m).to(torch.uint8))
                masks_t = torch.stack(mask_tensors, dim=0)
            else:
                masks_t = torch.empty((0, D, D), dtype=torch.uint8)
            labels["masks"] = masks_t

        if self.get_features:
            # example: load first caption feature if exists
            # keep same naming convention you used earlier
            feat_path = os.path.join(self.cache_path, "coco", "captions", f"{image_id}_0.pt")
            if os.path.exists(feat_path):
                feature = torch.load(feat_path).detach()
                return int(image_id), image_t, [feature]
            return int(image_id), image_t, []


        return int(image_id), image_t, labels

    def coco_collate_fn(self, batch):
        ids = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch], dim=0)

        cls_list = []
        bbox_list = []
        batch_idx_list = []
        masks_list = []

        for batch_id, item in enumerate(batch):
            lab = item[2]
            n = lab["cls"].shape[0] if lab["cls"].numel() else 0
            if n > 0:
                cls_list.append(lab["cls"])
                bbox_list.append(lab["bboxes"])
                batch_idx_list.append(torch.full((n,), batch_id, dtype=torch.long))
            if "masks" in lab:
                masks_list.append(lab["masks"])  # (n_i, H, W) or (0, H, W)

        labels = {}
        labels["cls"] = torch.cat(cls_list, dim=0) if cls_list else torch.empty((0,1), dtype=torch.long)
        labels["bboxes"] = torch.cat(bbox_list, dim=0) if bbox_list else torch.empty((0,4), dtype=torch.float32)
        labels["batch_idx"] = torch.cat(batch_idx_list, dim=0) if batch_idx_list else torch.empty((0,), dtype=torch.long)

        if masks_list:
            labels["masks"] = torch.cat(masks_list, dim=0)
        else:
            labels["masks"] = torch.empty((0, images.shape[2], images.shape[3]), dtype=torch.uint8)

        return ids, images, labels

    
    def collate_fn(self, batch):
        if self.get_features:
            images = torch.stack([item[1] for item in batch])
            ids = [item[0] for item in batch]
            features = [item[2] for item in batch]
            return ids, images, features
        else: 

            return self.coco_collate_fn(batch)


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
