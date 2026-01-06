import hashlib
import json
import logging
import os
import sqlite3
import multiprocessing
from collections import defaultdict
from typing import Optional

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
    # Get process ID for multiprocessing logging
    process_id = multiprocessing.current_process().pid
    
    # Create a logger with process-specific name to avoid conflicts
    logger = logging.getLogger(f"process_{process_id}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Clear any existing handlers to avoid duplicate logs
    logger.handlers.clear()
    
    # Create a new handler for this process
    handler = logging.StreamHandler()
    
    # Include process ID in the log format to distinguish between processes
    formatter = logging.Formatter(f"PID {process_id} - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def load_embeddings(db_path, query: Optional[str] = None):
    conn = sqlite3.connect(db_path)
    print("Loading embeddings from", db_path)
    table_name = db_path.split("/")[-1].split(".")[0]
    if table_name in ["umap", "tsne"]:
        query = query.split(".")
        model_name = query[0].split("_")[-1]
        query = "_".join([model_name, query[2]])
        df = pd.read_sql_query(
            f"SELECT run_id FROM embeddings ORDER BY id",
            conn,
        )
        df = pd.read_sql_query(
            f"SELECT x,y FROM embeddings WHERE run_id='{query}' ORDER BY id",
            conn,
        )
        conn.close()
        df.pop("run_id")
        return df[["x", "y"]].values
    else:
        if query is None:
            df = pd.read_sql_query("SELECT * FROM embeddings ORDER BY img_id", conn)
            df["embedding"] = df["embedding"].apply(
                lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
            )
            conn.close()
            print(len(df["img_id"].values))
            return (
                df["img_id"].values,
                np.concatenate(df["embedding"].values),
                df["hit_freq"].values,
                df["mean_iou"].values,
                df["mean_conf"].values,
                df["flag_cat"].values,
                df["flag_supercat"].values,
            )
        else:
            df = pd.read_sql_query(
                f"SELECT {query} FROM embeddings ORDER BY img_id", conn
            )
            if query == "embedding":
                df["embedding"] = df["embedding"].apply(
                    lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
                )
            conn.close()
            return df


def load_distances(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT i, j, distance FROM distances ORDER BY i, j", conn)
    conn.close()
    rows = df[["i", "j", "distance"]].values  # or cursor.fetchall() if you prefer
    if rows.size == 0:
        # No distances in table -> return empty matrix
        print("Distance table empty: returning (0,0) matrix")
        return np.zeros((0, 0), dtype=float)

    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]

    # Map original IDs to 0..n-1
    unique_ids = np.unique(np.concatenate([rows[:, 0], rows[:, 1]]))
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    n = len(unique_ids)
    print(f"Number of nodes: {n}")
    dist_matrix = np.zeros((n, n), dtype=float)

    # Safe remapping without relying on np.vectorize (fails on size 0 inputs)
    i_idx = np.array([id_to_idx.get(x) for x in rows[:, 0]], dtype=int)
    j_idx = np.array([id_to_idx.get(x) for x in rows[:, 1]], dtype=int)
    d_vals = rows[:, 2].astype(float)

    dist_matrix[i_idx, j_idx] = d_vals
    dist_matrix[j_idx, i_idx] = d_vals
    return dist_matrix
