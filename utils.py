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
    # Open DB in read-only mode to avoid creating journal/wal files on
    # read-only or network-mounted filesystems which can cause "disk I/O error".
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    except Exception:
        # Fallback to default (may raise same error which will be handled by caller)
        conn = sqlite3.connect(db_path)
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
            f"SELECT vector FROM embeddings WHERE run_id='{query}' ORDER BY id",
            conn,
        )
        df["vector"] = df["vector"].apply(
            lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
        )
        conn.close()
        # convert array-of-1xD rows into (n, D) matrix
        return np.concatenate(df["vector"].values, axis=0)
    else:
        if query is None:
            df = pd.read_sql_query("SELECT * FROM embeddings ORDER BY img_id", conn)
            df["embedding"] = df["embedding"].apply(
                lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
            )
            conn.close()
            # Build losses dict (may be missing for some runs)
            losses = {}
            for key in ("box_loss", "cls_loss", "dfl_loss", "seg_loss"):
                losses[key] = df[key].values if key in df.columns else None

            return (
                df["img_id"].values,
                np.concatenate(df["embedding"].values),
                df["hit_freq"].values if "hit_freq" in df.columns else None,
                df["mean_iou"].values if "mean_iou" in df.columns else None,
                df["mean_conf"].values if "mean_conf" in df.columns else None,
                df["flag_cat"].values if "flag_cat" in df.columns else None,
                df["flag_supercat"].values if "flag_supercat" in df.columns else None,
                losses,
            )
        else:
            df = pd.read_sql_query(
                f"SELECT {query} FROM embeddings ORDER BY img_id", conn
            )
            if query == "embedding":
                df["embedding"] = df["embedding"].apply(
                    lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
                )
                # If embedding column requested, return (n, D) matrix
                conn.close()
                return np.concatenate(df["embedding"].values, axis=0)
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
    # attempt common variants and support per-component directed distances (i, j, component, distance)
    df = None
    try:
        df = pd.read_sql_query("SELECT i, j, component, distance FROM distances ORDER BY component, i, j", conn)
        has_component = True
    except Exception:
        has_component = False
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

    # If component column is present, build a dict of directed matrices (no symmetrization)
    if 'component' in df.columns:
        # Group by component
        comps = sorted(df['component'].unique())
        # Determine unique ids across all rows
        unique_ids = np.unique(np.concatenate([df['i'].values, df['j'].values]))
        id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
        n = len(unique_ids)
        print(f"Number of nodes: {n} (from {db_path})")
        comp_mats = {}
        for comp in comps:
            comp_df = df[df['component'] == comp]
            mat = np.full((n, n), np.nan, dtype=float)
            for _, row in comp_df.iterrows():
                i = int(row['i'])
                j = int(row['j'])
                d = float(row['distance'])
                mat[id_to_idx[i], id_to_idx[j]] = d
            comp_mats[comp] = mat
        # If there's only a single component, return the matrix directly for backward compatibility
        if len(comp_mats) == 1:
            return list(comp_mats.values())[0]
        return comp_mats

    # legacy behavior: symmetric single distance column
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

    # Safe remapping without relying on np.vectorize (fails on size 0 inputs)
    i_idx = np.array([id_to_idx.get(x) for x in rows[:, 0]], dtype=int)
    j_idx = np.array([id_to_idx.get(x) for x in rows[:, 1]], dtype=int)
    d_vals = rows[:, 2].astype(float)

    dist_matrix[i_idx, j_idx] = d_vals
    dist_matrix[j_idx, i_idx] = d_vals
    return dist_matrix

# -----------------------------
# Hyperparameter query helpers
# -----------------------------

def parse_hyperparam_query(query: str):
    """
    Parse a simple hyperparameter query string into a list of filters.

    Supported examples:
      "n_neighbors=20,min_dist>0.1"
      "perplexity:40"
      "min_dist>=0.1"

    Returns:
      list of (key, op, value) tuples where op is one of ('>=','<=','>','<','=','==',':')
    """
    if not query:
        return []

    # Normalize separators and split tokens
    tokens = [t.strip() for t in query.replace(";", ",").split(",") if t.strip()]

    ops = [">=", "<=", ">", "<", "==", "=", ":"]
    filters = []

    for token in tokens:
        matched = False
        for op in ops:
            if op in token:
                key, val = token.split(op, 1)
                key = key.strip()
                val = val.strip()
                # Try to interpret value as JSON (so numbers/bools/null become native types)
                try:
                    val_parsed = json.loads(val)
                except Exception:
                    # Fallback: try float then keep as string
                    try:
                        val_parsed = float(val)
                    except Exception:
                        val_parsed = val
                filters.append((key, op, val_parsed))
                matched = True
                break
        if not matched:
            # No operator: treat as existence/equality true
            filters.append((token, "=", True))

    return filters


def match_hyperparams(params: dict, filters):
    """Return True if params dict satisfies all filters produced by parse_hyperparam_query."""
    if not filters:
        return True

    for key, op, val in filters:
        if key not in params:
            return False
        pval = params[key]

        # Coerce numeric-like strings to numbers for comparisons when possible
        try:
            if isinstance(pval, str) and pval.replace(".", "", 1).isdigit():
                pval_num = float(pval)
            else:
                pval_num = pval
        except Exception:
            pval_num = pval

        # Equality
        if op in ("=", "==", ":"):
            if pval != val:
                return False
        else:
            # For inequality operators, attempt numeric comparison
            try:
                if op == ">":
                    if not (float(pval_num) > float(val)):
                        return False
                elif op == "<":
                    if not (float(pval_num) < float(val)):
                        return False
                elif op == ">=":
                    if not (float(pval_num) >= float(val)):
                        return False
                elif op == "<=":
                    if not (float(pval_num) <= float(val)):
                        return False
            except Exception:
                return False

    return True


# -----------------------------
# Model grouping & color helpers
# -----------------------------

def parse_model_group(model_full: str):
    """Parse a model string into (base_model, distance_metric_or_suffix).

    Examples:
      'resnet50_l2' -> ('resnet50', 'l2')
      'yolov8s-seg' -> ('yolov8s-seg', None)
      'clip' -> ('clip', None)
    The heuristic splits on the LAST underscore. If no underscore, returns the full name as base.
    """
    if not isinstance(model_full, str):
        return model_full, None
    if "_" in model_full:
        base, suffix = model_full.rsplit("_", 1)
        return base, suffix
    return model_full, None


def _hsl_to_hex(h: float, s: float = 60.0, l: float = 55.0):
    """Convert HSL values (degrees, percent, percent) to hex color string."""
    # convert to [0,1]
    h = float(h) % 360.0
    s = float(s) / 100.0
    l = float(l) / 100.0

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = l - c / 2

    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0
    elif 120 <= h < 180:
        r1, g1, b1 = 0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x

    r = int((r1 + m) * 255)
    g = int((g1 + m) * 255)
    b = int((b1 + m) * 255)

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def color_for_group(group_name: str):
    """Deterministically map a group name to a hex color.

    Uses a hash of the group name to choose a hue (0-360). Returns hex string.
    """
    if group_name is None:
        return "#888888"
    # stable hash
    try:
        h = int(hashlib.sha256(group_name.encode()).hexdigest()[:8], 16) % 360
    except Exception:
        h = abs(hash(group_name)) % 360
    return _hsl_to_hex(h, s=60.0, l=55.0)
