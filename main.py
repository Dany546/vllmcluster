import argparse
import gc
import os
import random
import sqlite3
import sys

import wandb


def cluster(args):
    import numpy as np
    import pandas as pd
    import torch
    from cfg import augmentations, device, model_ckpt_path
    from clustering import Clustering_layer
    from dataset import COCODataset
    from model import DINO
    from torch.utils.data import Sampler
    from tqdm import tqdm

    class ResumeSampler(Sampler):
        def __init__(self, data_source, start_idx=0):
            self.data_source = data_source
            self.start_idx = start_idx
            random.seed(42)
            torch.manual_seed(42)

        def __iter__(self):
            return iter(range(self.start_idx, len(self.data_source)))

        def __len__(self):
            return len(self.data_source)  #  - self.start_idx

    if args.model is None:
        models = [f"yolo{i}{j}.pt" for i in ["v8", "12"] for j in ["s", "x"]]
    elif args.model == "all":
        models = [
            "yolov8s-seg.pt",
            "yolov8x-seg.pt",
            "yolo11s-seg.pt",
            "yolo11x-seg.pt",
            "yolov8s.pt",
            "yolov8x.pt",
            "yolo12s.pt",
            "yolo12x.pt",
            "dino",
            "dino_attention",
            "dinov3",
            "dinov3_attention",
            "clip",
        ]
    else:
        models = args.model.split(",")

    run = None
    if not args.debug and False:
        run = wandb.init(
            entity="miro-unet",
            project="VLLM clustering",
            # mode="offline",
            name=f"embeddings - {args.model if args.model else 'yolo'}",  # optional descriptive name
            config={
                "batch_size": 128,
                "lr": 1e-3,
                "weight_decay": 1e-3,
                "optimizer": "AdamW",
                "patch_size": 16,
            },
        )
    for model in models:
        dataset = COCODataset(
            data_split="validation",
            transform=augmentations["yolo" if "yolo" in model else model.replace("_attention", "").replace("v3", "")],
            caching=False,
            get_features=False,
            segmentation='seg' in model,
        )
        torch.manual_seed(42)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            prefetch_factor=2,
            sampler=ResumeSampler(dataset),
            collate_fn=dataset.collate_fn,
        )
        clustering_layer = Clustering_layer(
            model,
            model.split(".")[0],
            debug=args.debug,
            store_individual_predictions=args.store_preds,
        )

        # Optionally save raw YOLO heads into the embeddings DB first
        if getattr(args, "save_raw", False) and "yolo" in model:
            conn = sqlite3.connect(clustering_layer.embeddings_db, timeout=30)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS predictions_raw_heads (
                        id INTEGER PRIMARY KEY,
                        img_id INTEGER NOT NULL,
                        model_name TEXT NOT NULL,
                        head_blob BLOB NOT NULL,
                        head_shapes TEXT NOT NULL,
                        head_dtypes TEXT NOT NULL,
                        anchors_info TEXT,
                        features BLOB,
                        created_ts INTEGER,
                        UNIQUE(img_id, model_name)
                    )"""
            )
            conn.commit()
            for batch_idx, (img_ids, images, labels) in enumerate(dataloader):
                images = images.to(device)
                clustering_layer.model.predict_and_store_raw(
                    images, img_ids, emb_conn=conn, model_name=clustering_layer.model_name
                )
            conn.close()
            print(f"Saved raw heads for {clustering_layer.model_name} to {clustering_layer.embeddings_db}")
            if not getattr(args, "cluster", False):
                continue

        clustering_layer.distance_matrix_db(dataloader)

        if run and False:
            conn = sqlite3.connect(clustering_layer.embeddings_db)
            df = pd.read_sql_query("SELECT * FROM embeddings", conn)
            df["embedding"] = df["embedding"].apply(
                lambda b: np.frombuffer(b, dtype=np.float32).tolist()
            )
            conn.close()
            table = wandb.Table(dataframe=df)
            run.log({f"embeddings_{model}": table})
            df = None
    if run:
        run.finish()


def main(args):
    name = "cluster; " if args.cluster else ""
    name += "visu; " if args.visu else ""
    name += "knn" if args.knn else ""
    wandb.init(
        entity="miro-unet",
        project="VLLM clustering",
        # mode="offline",
        name=name,  # optional descriptive name
    )
    if args.visu:
        # from cluster_visualization import visualize_clusters
        from project import project

        # visualize_clusters(args)
        project(args)
    elif args.knn:
        from evaluate_clusters import KNN

        KNN(args)
    else: 
        cluster(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode without wandb logging"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="comma separated list of models to use for embedding, defaults to several yolo models",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Compute embeddings and distance matrice",
    )
    parser.add_argument(
        "--visu",
        action="store_true",
        help="Visualize the clusters",
    )
    parser.add_argument(
        "--knn",
        action="store_true",
        help="Train KNN for conformable predictions",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Comma-separated list of table basenames or full paths to run KNN on (optional). If provided, only these tables will be processed.",
    )
    parser.add_argument(
        "--model_filter",
        default=None,
        help="Comma-separated substrings to filter which model pairs to run KNN on (e.g., 'yolov11x-seg')",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated list of targets to run (e.g., 'mean_iou,cls_loss,seg_loss')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="perform all steps, will override other flags if set",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Run model to save raw YOLO head predictions into embeddings DB",
    )
    parser.add_argument(
        "--store-preds",
        action="store_true",
        help="Store per-detection predictions (bboxes/masks) in the embeddings DB (default: off)",
    )
    args = parser.parse_args()
    args.all = (not (args.cluster and args.visu and args.knn)) | args.all
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
