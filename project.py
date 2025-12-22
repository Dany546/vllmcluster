import hashlib
import itertools
import json
import logging
import os
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from utils import dict_to_filename, get_logger, get_lookups


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


def compute_run_id(model: str, hyperparams: dict) -> str:
    """
    Compute a unique run_id from model name and hyperparameters.
    Deterministic: same inputs -> same run_id.
    """
    # Sort hyperparams to ensure consistent ordering
    hp_items = sorted(hyperparams.items())
    hp_str = json.dumps(hp_items)  # stable string representation

    # Build base string
    base = f"{model}:{hp_str}"

    # Hash to short identifier
    run_hash = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]

    # Final run_id
    run_id = f"{model}_{run_hash}"
    return run_id


def compute_and_store(X, model_name, hyperparams, db_path="", algo="tsne"):
    assert algo in ["tsne", "umap"], "Unknown algorithm"

    # Compute run_id
    run_id = compute_run_id(model_name, hyperparams)

    # Instantiate projector
    if algo == "tsne":
        projector = TSNE(**hyperparams, n_components=2, random_state=42)
    else:
        projector = UMAP(**hyperparams, n_components=2, random_state=42)

    embedding = projector.fit_transform(X)

    # Connect to DB
    conn = sqlite3.connect(os.path.join(db_path, f"{algo}.db"))
    cur = conn.cursor()

    # Insert metadata
    cols = ", ".join(["model"] + list(hyperparams.keys()))
    placeholders = ", ".join(
        ["?"] * (len(hyperparams) + 1 + 1)
    )  # run_id + model + hyperparams
    sql = f"INSERT OR REPLACE INTO metadata(run_id, {cols}) VALUES ({placeholders})"
    cur.execute(sql, (run_id, model_name, *hyperparams.values()))
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            x FLOAT,
            y FLOAT,
            FOREIGN KEY(run_id) REFERENCES metadata(run_id)
        )
    """)
    # Insert embeddings
    for x, y in embedding:
        cur.execute(
            "INSERT INTO embeddings(run_id, x, y) VALUES (?, ?, ?)",
            (run_id, float(x), float(y)),
        )

    conn.commit()
    conn.close()

    return run_id


def compute_and_store_metrics(model_name, data, db_path=""):
    anns_by_image, id_to_name, id_to_super, categories, supercategories = get_lookups()
    ids, hfs, mious, mconfs, cats, supercats = data

    conn = sqlite3.connect(os.path.join(db_path, f"metrics.db"))
    cur = conn.cursor()

    # Ensure table exists with dynamic category/supercategory columns
    cat_cols = ", ".join([f'"{c}" INT' for c in categories])
    super_cols = ", ".join([f'"{sc}" INT' for sc in supercategories])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            img_id TEXT,
            hit_freq FLOAT,
            mean_iou FLOAT,
            mean_conf FLOAT,
            {cat_cols},
            {super_cols},
            FOREIGN KEY(model) REFERENCES metadata(model)
        )
    """)

    # Iterate over images
    for img_id, hf, miou, mconf, cat, supcat in zip(
        ids, hfs, mious, mconfs, cats, supercats
    ):
        anns = anns_by_image[img_id]

        # Initialize counts
        cat_counts = {name: 0 for name in categories}
        super_counts = {sc: 0 for sc in supercategories}

        # Count instances
        for ann in anns:
            cat_name = id_to_name[ann["category_id"]]
            supercat = id_to_super[ann["category_id"]]
            cat_counts[cat_name] += 1
            super_counts[supercat] += 1

        # Build row values
        values = [model_name, img_id, hf, miou, mconf]
        values += list(cat_counts.values())
        values += list(super_counts.values())

        placeholders = ", ".join(["?"] * len(values))
        cur.execute(
            f"""
            INSERT INTO metrics(model, img_id, hit_freq, mean_iou, mean_conf,
                                {", ".join(categories)},
                                {", ".join(supercategories)})
            VALUES ({placeholders})
        """,
            values,
        )

    conn.commit()
    conn.close()


def project(args):
    db_path = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, f) for f in os.listdir(db_path) if f.endswith(".db")
    ]
    logger = get_logger(args.debug)

    db_path = "/globalscratch/ucl/irec/darimez/dino/"
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

    # Build hyperparam combinations
    def expand_params(param_dict, algo):
        keys = list(param_dict.keys())
        for values in itertools.product(*[param_dict[k] for k in keys]):
            d = dict(zip(keys, values))
            d["algo"] = algo
            yield d

    all_hyperparams = list(expand_params(umap_params, "umap")) + list(
        expand_params(tsne_params, "tsne")
    )

    for table in tables:
        model_name = os.path.basename(table).split(".")[0]
        logger.debug(f"Processing {model_name}")

        ids = X = hfs = mious = mconfs = cats = supercats = None
        # Check if run_id already exists in metadata
        conn = sqlite3.connect(os.path.join(db_path, f"metrics.db"))
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1 FROM metrics WHERE model=?", (model_name,))
            exists = cur.fetchone()
        except Exception as e:
            exists = False
        conn.close()
        if exists:
            logger.debug(f"Loading existing metrics for model {model_name}")
            continue
        else:
            logger.debug(f"Saving metrics for model {model_name}")
            ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
            data = (ids, hfs, mious, mconfs, cats, supercats)
            compute_and_store_metrics(model_name, data, db_path=db_path)

        for hyperparam in all_hyperparams:
            algo = hyperparam["algo"]
            run_id = compute_run_id(
                model_name, {k: v for k, v in hyperparam.items() if k != "algo"}
            )

            # Check if run_id already exists in metadata
            conn = sqlite3.connect(os.path.join(db_path, f"{algo}.db"))
            cur = conn.cursor()
            cols = ", ".join(
                ["model TEXT"] + list(str(h) + " REAL" for h in hyperparam.keys())
            )
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS metadata (
                    run_id TEXT,
                    {cols},
                )
            """)
            conn.commit()
            cur.execute("SELECT 1 FROM metadata WHERE run_id=?", (run_id,))
            exists = cur.fetchone()
            conn.close()
            if exists:
                logger.debug(f"Skipping existing run {run_id}")
                continue
            else:
                # Compute and store embeddings
                logger.debug(f"Computing ...")
                if X is None:
                    ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
                compute_and_store(
                    X,
                    model_name,
                    {k: v for k, v in hyperparam.items() if k != "algo"},
                    db_path=db_path,
                    algo=algo,
                )
