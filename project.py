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
from utils import dict_to_filename, get_logger, get_lookups, load_embeddings


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
        projector = TSNE(**hyperparams, random_state=42)
    else:
        projector = UMAP(**hyperparams, random_state=42)

    embedding = projector.fit_transform(X)

    # Connect to DB
    conn = sqlite3.connect(os.path.join(db_path, f"{algo}.db"))
    cur = conn.cursor()

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
    rows = []
    for i, (x, y) in enumerate(embedding):
        if len(rows) >= 128:
            cur.executemany(
                "INSERT OR IGNORE INTO embeddings(id, run_id, x, y) VALUES (?, ?, ?, ?)",
                rows,
            )
            rows = []
            conn.commit()
        else:
            rows.append((int(i), run_id, float(x), float(y)))

    if len(rows) > 0:
        query = (
            """INSERT OR IGNORE INTO embeddings(id, run_id, x, y) VALUES (?, ?, ?, ?)"""
        )
        cur.executemany(query, rows)
        conn.commit()

    # Insert metadata
    cols = ", ".join(["model"] + list(hyperparams.keys()))
    placeholders = ", ".join(
        ["?"] * (len(hyperparams) + 1 + 1)
    )  # run_id + model + hyperparams
    sql = f"INSERT INTO metadata(run_id, {cols}) VALUES ({placeholders})"
    cur.execute(sql, (run_id, model_name, *hyperparams.values()))

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
            model TEXT,
            img_id INTEGER,
            hit_freq FLOAT,
            mean_iou FLOAT,
            mean_conf FLOAT,
            {cat_cols},
            {super_cols},
            FOREIGN KEY(model) REFERENCES metadata(model),
            PRIMARY KEY (model, img_id)
        )
    """)

    cat_cols = cat_cols.replace(" INT", "")
    super_cols = super_cols.replace(" INT", "")
    placeholders = ", ".join(["?"] * (len(categories) + len(supercategories) + 5))
    rows = []
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
        values = [model_name, int(img_id), float(hf), float(miou), float(mconf)]
        values += list(map(int, cat_counts.values()))
        values += list(map(int, super_counts.values()))

        if len(rows) >= 128:
            query = f"""
            INSERT OR IGNORE INTO metrics(model, img_id, hit_freq, mean_iou, mean_conf,
                                {cat_cols}, {super_cols}) VALUES ({placeholders})
            """
            cur.executemany(query, rows)
            rows = []
            conn.commit()
        else:
            rows.append(values)

    if len(rows) > 0:
        query = f"""INSERT OR IGNORE INTO metrics(model, img_id, hit_freq, mean_iou, mean_conf,
                            {cat_cols}, {super_cols}) VALUES ({placeholders})"""
        cur.executemany(query, rows)
        conn.commit()
    conn.close()


def get_tasks(all_hyperparams, model_name):
    tasks = []
    for hyperparam in all_hyperparams:
        algo = hyperparam["algo"]
        run_id = compute_run_id(
            model_name, {k: v for k, v in hyperparam.items() if k != "algo"}
        )
        # Check if run_id already exists in metadata
        conn = sqlite3.connect(
            os.path.join("/globalscratch/ucl/irec/darimez/dino/proj/", f"{algo}.db")
        )
        cur = conn.cursor()
        cols = ", ".join(
            ["model TEXT"] + list(str(h) + " REAL" for h in hyperparam.keys())
        )
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS metadata (
                run_id TEXT,
                {cols}
            )
        """)
        conn.commit()
        cur.execute("SELECT 1 FROM metadata WHERE run_id=?", (run_id,))
        exists = cur.fetchone()
        conn.close()
        if exists is None:
            tasks.append(hyperparam)
    return tasks


def project(args):
    db_path = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, f) for f in os.listdir(db_path) if f.endswith(".db")
    ]
    logger = get_logger(args.debug)

    db_path = "/globalscratch/ucl/irec/darimez/dino/proj/"
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
        "max_iter": [1000],
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
    # deletes failed entries from embeddings
    for db in ["tsne.db", "umap.db"]:
        conn = sqlite3.connect(
            os.path.join("/globalscratch/ucl/irec/darimez/dino/proj/", db)
        )
        c = conn.cursor()
        try:
            c.execute(
                """ DELETE FROM metadata WHERE run_id NOT IN ( SELECT DISTINCT run_id FROM embeddings ); """
            )
        except Exception as e:
            print(e)
        conn.commit()
        conn.close()

    for table in tables:
        model_name = os.path.basename(table).split(".")[0]
        logger.debug(f"Processing {model_name}")

        ids = X = hfs = mious = mconfs = cats = supercats = None
        # Check if run_id already exists in metadata
        conn = sqlite3.connect(os.path.join(db_path, f"metrics.db"))
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM metrics WHERE model=?", (model_name,))
            exists = cur.fetchone()[0] == 5000
        except Exception as e:
            exists = False
        conn.close()
        if exists:
            logger.info(f"Loading existing metrics for model {model_name}")
            continue
        else:
            logger.info(f"Saving metrics for model {model_name}")
            ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
            data = (ids, hfs, mious, mconfs, cats, supercats)
            compute_and_store_metrics(model_name, data, db_path=db_path)

        tasks = get_tasks(all_hyperparams, model_name)
        logger.info(
            f"Computing projections for {len(tasks)}/{len(all_hyperparams)} parameters"
        )
        for task in tasks:
            if X is None:
                ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
            compute_and_store(
                X,
                model_name,
                {k: v for k, v in task.items() if k != "algo"},
                db_path=db_path,
                algo=task["algo"],
            )
