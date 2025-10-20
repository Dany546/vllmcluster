from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, data_split, transform=None, caching=False):
        self.cache_path = "//DINO"
        self.dataset = load_dataset("coco_captions", split=data_split)
        if caching:
            cache_features(self)
        self.transform = transform
        self.clip = self.clip_tokenizer = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        caption = self.dataset[idx]["caption"]
        if os.path.exists(f"{self.cache_path}/{self.dataset[idx]['id']}.pt"):
            features = torch.load(f"{self.cache_path}/{self.dataset[idx]['id']}.pt")
        else:
            if self.clip is None:
                self.clip = CLIPModel.from_pretrained(
                    f"openai/clip-vit-base-patch{patch_size}"
                ).text_model.to(device)
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                    f"openai/clip-vit-base-patch{patch_size}"
                ).to(device)
            features = self.clip_tokenizer(caption, return_tensors="pt").input_ids.to(
                device
            )
            torch.save(features, f"{self.cache_path}/{self.dataset[idx]['id']}.pt")

        if self.transform:
            image = self.transform(image)

        return image, features


if __name__ == "__main__":
    # Load CLIP model
    clip = CLIPModel.from_pretrained(
        f"openai/clip-vit-base-patch{patch_size}"
    ).text_model.to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        f"openai/clip-vit-base-patch{patch_size}"
    ).to(device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    def cache_features(dataset):
        for data in dataset.dataset:
            # image = Image.fromarray(data["image"])
            # for aug in augmentations:
            #     with torch.no_grad():
            #         feature = dino(aug(image).unsqueeze(0)).squeeze()
            #         feature = get_visual_embeddings(feature, model.attention_weights)
            #         torch.save(features, cache_path)
            with torch.no_grad():
                feature = clip_tokenizer(data["caption"]).to(device)
                feature = clip(feature)
                torch.save(features, f"{dataset.cache_path}/{data['id']}.pt")
