import hashlib
import json
import logging
import os
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd


def get_lookups():
    with open(
        "/globalscratch/ucl/irec/darimez/dino/coco/validation/annotations/instances_val2017.json",
        "r",
    ) as f:
        data = json.load(f)

    # --- Build lookup maps ---
    id_to_name = {
        cat["id"]: cat["supercategory"] + "_" + cat["name"]
        for cat in data["categories"]
    }
    id_to_super = {cat["id"]: cat["supercategory"] for cat in data["categories"]}

    # Define category and supercategory lists
    categories = list(set(id_to_name.values()))
    supercategories = list(set(id_to_super.values()))

    # --- Group annotations by image ---
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    return anns_by_image, id_to_name, id_to_super, categories, supercategories


def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def dict_to_filename(d):
    # Convert dict to a stable representation: sorted items
    items = tuple(sorted(d.items()))
    # Hash it
    h = hashlib.sha256(str(items).encode()).hexdigest()
    # Shorten for filename
    return h[:12]  # first 12 hex chars


def get_logger(debug):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_embeddings(db_path):
    conn = sqlite3.connect(db_path)
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


def load_distances(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT i, j, distance FROM distances", conn)
    conn.close()
    rows = df[["i", "j", "distance"]].values  # or cursor.fetchall() if you prefer
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]

    # Map original IDs to 0..n-1
    unique_ids = np.unique(np.concatenate([rows[:, 0], rows[:, 1]]))
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    n = len(unique_ids)
    print(f"Number of nodes: {n}")
    dist_matrix = np.zeros((n, n), dtype=float)
    # Vectorized remapping
    i_idx = np.vectorize(id_to_idx.get)(rows[:, 0])
    j_idx = np.vectorize(id_to_idx.get)(rows[:, 1])
    d_vals = rows[:, 2]
    dist_matrix[i_idx, j_idx] = d_vals
    dist_matrix[j_idx, i_idx] = d_vals
    return dist_matrix
