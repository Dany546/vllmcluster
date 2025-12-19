import itertools
import json
import logging
import os
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
from plotting import log_plot_plotly as old_log_plot_plotly
from sklearn.manifold import TSNE
from sql_plotting import log_plot_plotly
from umap import UMAP
from utils import dict_to_filename, get_lookups, table_exists

import wandb


# Load embeddings from SQL
def load_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    df = pd.read_sql_query("SELECT * FROM embeddings", conn)
    df["embedding"] = df["embedding"].apply(
        lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
    )
    # df["error"] = df["error"].apply(
    #     lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
    # )
    conn.close()
    return (
        df["img_id"].values,
        np.concatenate(df["embedding"].values),
        df["hit_freq"].values,
        df["mean_iou"].values,
        df["mean_conf"].values,
        df["flag_cat"].values,
        df["flag_supercat"].values,
    )


def log_plot(embeddings, labels, label_names, sizes, title, run):
    # Keep reference to scatter for colorbar
    scatter = None
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    for embedding, ax, ax_title in zip(embeddings, axes, ["UMAP", "t-SNE"]):
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab20",
            s=20,
        )
        ax.set_title(ax_title)
    # Adjust layout to make room for colorbar
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cbar = fig.colorbar(
        scatter,
        ax=axes,
        orientation="vertical",
        ticks=np.arange(len(label_names)),
    )
    cbar.ax.set_yticklabels(label_names)
    run.log({f"{title} plot": wandb.Image(fig)})
    plt.close()


def get_logger(debug):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def visualize_clusters(args):
    # -----------------------------
    # Loop over multiple models/tables
    # -----------------------------
    db_path = f"/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db")
    ]  # extend as needed
    logger = get_logger(args.debug)
    if not args.debug:
        if wandb.run is None:
            # No run yet → initialize a new one
            wandb.init(
                entity="miro-unet",
                project="VLLM clustering",
                # mode="offline",
                name=f"visu",  # optional descriptive name
            )
        run = wandb.run
    umap_params = {
        "n_neighbors": [15, 30, 60],
        "min_dist": [0.02, 0.1, 0.5],
        "n_components": [2],
    }
    tsne_params = {
        "n_components": [2],
        "perplexity": [10, 30, 50],
        "early_exaggeration": [8, 12, 16],
        "learning_rate": [100, 200, 500],
        "n_iter": [1000],
    }
    params = itertools.product(umap_params.values(), tsne_params.values())
    correlation = []
    for table in tables:
        logger.debug(table)
        ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)

        for umap_param, tsne_param in params:
            hyperparams = {
                "umap": umap_param,
                "tsne": tsne_param,
            }
            sql_names = [
                dict_to_filename(hyperparams["umap"]),
                dict_to_filename(hyperparams["tsne"]),
            ]
            emb_conn = sqlite3.connect(table.replace("embeddings", "proj"), timeout=30)
            emb_conn.execute("PRAGMA journal_mode=WAL")
            emb_conn.execute("PRAGMA synchronous=OFF")
            emb_conn.execute("PRAGMA temp_store=MEMORY")
            emb_conn.execute("PRAGMA cache_size=-20000")
            emb_cursor = emb_conn.cursor()
            if not table_exists(emb_cursor, f"umap_{sql_names[0]}"):
                sql_columns = ", ".join(
                    [
                        "comp_" + str(i) + " REAL"
                        for i in range(hyperparams["umap"]["n_components"])
                    ]
                )
                emb_cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS umap_{sql_names[0]} (
                        id INTEGER PRIMARY KEY
                        {sql_columns}
                    )""")
                umap = UMAP(**hyperparams["umap"])
                embedding_umap = umap.fit_transform(X)
            else:
                embedding_umap = np.zeros((len(X), 2))
            if not table_exists(emb_cursor, f"tsne_{sql_names[1]}"):
                sql_columns = ", ".join(
                    [
                        "comp_" + str(i) + " REAL"
                        for i in range(hyperparams["tsne"]["n_components"])
                    ]
                )
                emb_cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS tsne_{sql_names[1]} (
                        id INTEGER PRIMARY KEY
                        {sql_columns}
                    )""")
                tsne_gpu = TSNE(**hyperparams["tsne"])
                embedding_tsne = tsne_gpu.fit_transform(X)
            else:
                embedding_tsne = np.zeros((len(X), 2))
            # --- Load COCO JSON ---
            anns_by_image, id_to_name, id_to_super, categories, supercategories = (
                get_lookups()
            )
            rows = []
            umap_rows = [] if embedding_umap is not None else None
            tsne_rows = [] if embedding_tsne is not None else None
            for emb_umap, emb_tsne, hf, miou, mconf, cat, supercat, img_id in zip(
                embedding_umap, embedding_tsne, hfs, mious, mconfs, cats, supercats, ids
            ):
                anns = anns_by_image[img_id]  # from COCO JSON grouping

                # Initialize counts
                cat_counts = {name: 0 for name in categories}
                super_counts = {sc: 0 for sc in supercategories}

                # Count instances
                for ann in anns:
                    cat_name = id_to_name[ann["category_id"]]
                    supercat = id_to_super[ann["category_id"]]
                    cat_counts[cat_name] += 1
                    super_counts[supercat] += 1

                # Build row
                row = (
                    list(emb_umap)
                    + list(emb_tsne)
                    + list(cat_counts.values())
                    + list(super_counts.values())
                    + list([hf, miou, mconf])
                )
                rows.append(row)
                if umap_rows:
                    umap_rows.append([len(row)] + list(emb_umap))
                if tsne_rows:
                    tsne_rows.append([len(row)] + list(emb_tsne))
                if len(row) % 500 == 0:
                    if umap_rows:
                        emb_cursor.execute(
                            f"INSERT INTO umap_{sql_names[0]} VALUES ({','.join(['?' for _ in range(len(umap_rows[0]))])})",
                            umap_rows,
                        )
                        umap_rows = [] if embedding_umap is not None else None
                    if tsne_rows:
                        emb_cursor.execute(
                            f"INSERT INTO tsne_{sql_names[1]} VALUES ({','.join(['?' for _ in range(len(tsne_rows[0]))])})",
                            tsne_rows,
                        )
                        tsne_rows = [] if embedding_tsne is not None else None

                    emb_conn.commit()
            if umap_rows:
                emb_cursor.execute(
                    f"INSERT INTO umap_{sql_names[0]} VALUES ({','.join(['?' for _ in range(len(umap_rows[0]))])})",
                    umap_rows,
                )
                umap_cols = [f"UMAP-{i}" for i in range(embedding_umap.shape[1])]
            if tsne_rows:
                emb_cursor.execute(
                    f"INSERT INTO tsne_{sql_names[1]} VALUES ({','.join(['?' for _ in range(len(tsne_rows[0]))])})",
                    tsne_rows,
                )
                tsne_cols = [f"t-SNE-{i}" for i in range(embedding_tsne.shape[1])]
            emb_conn.commit()
            emb_conn.close()
            # Build dataframe
            # columns = (
            #     [f"UMAP-{i}" for i in range(embedding_umap.shape[1])]
            #     + [f"t-SNE-{i}" for i in range(embedding_tsne.shape[1])]
            #     + categories
            #     + supercategories
            #     + ["hit_freq", "mean_iou", "mean_conf"]
            # )
            # df = pd.DataFrame(rows, columns=columns)

            # Select only UMAP and t-SNE columns

            # Or just cross-correlations (UMAP vs t-SNE only)
            # cross_corr = df[umap_cols].corrwith(df[tsne_cols])
            # correlation.append(cross_corr)

        if run:
            # Log to W&B
            table_name = table.split(f"{os.sep}")[-1].split(".")[0]
            # run.log({f"{table_name}": wandb.Table(dataframe=df)})
            # log_plot(
            #     [embedding_umap, embedding_tsne],
            #     df[supercategories].values.argmax(1),
            #     supercategories,
            #     df["hit_freq"],
            #     f"{table_name}",
            #     run,
            # )
            fig = log_plot_plotly(
                df,
                umap_col=["UMAP-0", "UMAP-1"],
                tsne_col=["t-SNE-0", "t-SNE-1"],
                class_cols=categories,
                superclass_cols=supercategories,
                continuous_cols=["hit_freq", "mean_iou", "mean_conf"],
                title="Embedding Visualization",
            )
            run.log({f"{table_name}": fig})

    if run:
        correlation_df = pd.concat(correlation, axis=1)
        run.log({"correlation": wandb.Table(dataframe=correlation_df)})
        run.finish()
