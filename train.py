from dataset import COCODataset
from model import DINO
from clustering import Clustering_layer
from tqdm import tqdm
import wandb, torch, gc, os
from cfg import device, model_ckpt_path


def main():
    run = wandb.init(
        entity="miro-unet",
        project="VLLM clustering",
        # mode="offline",
        name="dino2clip_train",  # optional descriptive name
        config={
            "batch_size": 512,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "optimizer": "AdamW",
            "patch_size": 16,
        },
    )

    dataset = COCODataset(data_split="train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=run.config["batch_size"],
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    model = DINO().to(device).train()
    clustering_layer = Clustering_layer(100).to(device).train()

    # Simplified training loop for DINO and CLIP
    optimizer = torch.optim.AdamW(
        model.projection_layer.parameters(),
        lr=run.config["lr"],
        weight_decay=run.config["weight_decay"],
        betas=(0.9, 0.99),
    )

    accumulation_steps = int(512 / run.config["batch_size"])
    batch_size = accumulation_steps * run.config["batch_size"]

    for epoch in range(20):
        loss = 0
        pbar = tqdm(
            enumerate(dataloader),
            desc=f"Training Epoch {epoch + 1}",
            unit="batch",
            total=len(dataloader),
        )
        for batch_indx, (images, text_embeddings) in pbar:  # small subset for demo
            torch.cuda.empty_cache()
            gc.collect()

            out, loss_ = model(images.to(device), text_embeddings.to(device))
            loss += loss_ / accumulation_steps

            if (batch_indx + 1) % accumulation_steps == 0:
                loss.backward()
                optimizer.step()
                log_loss = loss.item() * accumulation_steps
                run.log({"loss": log_loss, "step": batch_indx})
                pbar.set_postfix({"loss": f"{log_loss:.3g}"})
                loss = 0
                optimizer.zero_grad()
        torch.save(
            {"epoch": epoch, "model": model.projection_layer.state_dict()},
            os.path.join(model_ckpt_path, f"model_dino.pth"),
        )
        if epoch % 2 == 0:
            clusters = clustering_layer(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=run.config["batch_size"],
                    shuffle=False,
                    num_workers=8,
                    drop_last=False,
                )
            )
            torch.save(
                {"epoch": epoch, "labels": clusters[0], "centers": clusters[1]},
                os.path.join(model_ckpt_path, f"clusters_dino_{epoch}.pth"),
            )


if __name__ == "__main__":
    main()
