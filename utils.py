import hashlib
import json
import logging
import os
import sqlite3
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
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
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
    # Support fallback locations: if the provided path does not exist, try
    # $CECIHOME/distances/<basename> and <basename>.db. Also accept databases
    # that use column name 'dist' instead of 'distance'.
    original_path = db_path
    if not os.path.exists(db_path):
        base = os.environ.get("CECIHOME", "/CECI/home/ucl/irec/darimez")
        alt = os.path.join(base, "distances", os.path.basename(db_path))
        if os.path.exists(alt):
            db_path = alt
        elif os.path.exists(alt + ".db"):
            db_path = alt + ".db"
        else:
            # try basename + .db directly
            alt2 = os.path.join(base, "distances", os.path.basename(db_path) + ".db")
            if os.path.exists(alt2):
                db_path = alt2

    conn = sqlite3.connect(db_path)
    # attempt common variants
    df = None
    try:
        df = pd.read_sql_query("SELECT i, j, distance FROM distances ORDER BY i, j", conn)
    except Exception:
        try:
            df = pd.read_sql_query("SELECT i, j, dist FROM distances ORDER BY i, j", conn)
        except Exception:
            # scan tables for a likely distances table
            try:
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                for t in tables['name'].values:
                    try:
                        info = pd.read_sql_query(f"PRAGMA table_info({t})", conn)
                        cols = info['name'].values.tolist()
                        if ('i' in cols or 'row_i' in cols) and ('j' in cols or 'row_j' in cols) and (
                            'distance' in cols or 'dist' in cols
                        ):
                            dist_col = 'distance' if 'distance' in cols else 'dist'
                            df = pd.read_sql_query(f"SELECT i, j, {dist_col} FROM {t} ORDER BY i, j", conn)
                            break
                    except Exception:
                        continue
            except Exception:
                df = None

    conn.close()
    if df is None or df.shape[0] == 0:
        raise RuntimeError(f"Could not read distances from {original_path} (resolved to {db_path})")

    # Normalize column name to 'distance'
    if 'distance' in df.columns:
        rows = df[["i", "j", "distance"]].values
    else:
        rows = df[["i", "j", "dist"]].values

    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]

    # Map original IDs to 0..n-1
    unique_ids = np.unique(np.concatenate([rows[:, 0], rows[:, 1]]))
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    n = len(unique_ids)
    print(f"Number of nodes: {n} (from {db_path})")
    dist_matrix = np.zeros((n, n), dtype=float)
    # Vectorized remapping
    i_idx = np.vectorize(id_to_idx.get)(rows[:, 0])
    j_idx = np.vectorize(id_to_idx.get)(rows[:, 1])
    d_vals = rows[:, 2]
    dist_matrix[i_idx, j_idx] = d_vals
    dist_matrix[j_idx, i_idx] = d_vals
    return dist_matrix
