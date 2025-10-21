from dataset import COCODataset
from model import DINO
from tqdm import tqdm
import wandb, torch, gc
from cfg import device, model_ckpt_path


def main():
    run = wandb.init(
        project="your_project_name",  # replace with your project name
        entity="your_team_name",  # replace with your W&B team name
        mode="offline",
        name="dino_training_offline",  # optional descriptive name
        config={
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "optimizer": "AdamW",
            "patch_size": 32,
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

    # Simplified training loop for DINO and CLIP
    optimizer = torch.optim.AdamW(
        model.projection_layer.parameters(),
        lr=run.config["lr"],
        weight_decay=run.config["weight_decay"],
        betas=(0.9, 0.99),
    )

    accumulation_steps = int(512 / run.config["batch_size"])
    batch_size = accumulation_steps * run.config["batch_size"]

    for epoch in range(100):
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
            continue
            out, loss_ = model(images.to(device), text_embeddings.to(device))
            loss += loss_ / accumulation_steps

            if (batch_indx + 1) % accumulation_steps == 0:
                loss.backward()
                optimizer.step()
                log_loss = loss.item() * accumulation_steps / batch_size
                run.log({"loss": log_loss, "step": batch_indx})
                pbar.set_postfix({"loss": f"{log_loss:.4f}"})
                loss = 0
                optimizer.zero_grad()
        break
        torch.save(
            model.projection_layer.state_dict(),
            os.path.join(model_ckpt_path, f"model_dino.pth"),
        )


if __name__ == "__main__":
    main()
