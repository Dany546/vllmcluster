"""
SQL-vector-aware KNN evaluation entrypoint.

This module provides a parallel implementation of the evaluation pipeline
that reads low-dimensional projections from vec_projections (sql-vector)
using the helpers in `utils_vec.py` and `proj_helpers_vec.py` while
reusing core evaluation helpers from `evaluate_clusters.py`.
"""
import os
import sqlite3
import itertools
from sklearn.model_selection import KFold
from typing import Dict

import numpy as np
import pandas as pd

from utils import get_logger, load_distances, load_embeddings
from utils_vec import load_projection_vectors
from proj_helpers_vec import list_proj_runs

# Reuse core helpers from the legacy evaluator to avoid duplication
from evaluate_clusters import (
    init_db,
    ensure_column_exists,
    already_computed,
    insert_record,
    work_fn,
    plot_results,
)

DB_PATH = "/globalscratch/ucl/irec/darimez/dino/knn_results.db"
EMBEDDINGS_DIR = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
PROJ_DB_DIR = "/globalscratch/ucl/irec/darimez/dino/proj/"


def get_lower_dim_tables():
    # Use proj_helpers_vec to enumerate vec_projections runs
    table_names, table_paths = list_proj_runs(PROJ_DB_DIR)
    return table_names, table_paths


def get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric):
    # Re-implement similar logic to original get_tasks but use proj runs
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    targets = ["hit_freq", "mean_iou", "flag_supercat"]
    n_components = [2, 10, 50]
    k_targ = list(itertools.product(neighbor_grid, targets, n_components))
    tasks: Dict = {}

    lower_dim_names, lower_dim_tables = get_lower_dim_tables()
    table_names = [os.path.basename(t).split(".")[0] for t in tables] + lower_dim_names
    tables = tables + lower_dim_tables

    for table, table_name in zip(tables, table_names):
        if ("clip" in table_name) or ("dino" in table_name):
            table_pairs = [
                f"{table_name}_{tn}"
                for tn in table_names
                if ("clip" not in tn) and ("dino" not in tn)
            ]
        else:
            table_pairs = [table_name]

        k_targ_per_table = itertools.product(k_targ, table_pairs)
        for (k, target, n_component), table_pair in k_targ_per_table:
            other_columns = {"target": target, "n_components": n_component}
            if not already_computed(table_pair, k, RN, num_folds, "kfold", distance_metric, other_columns, c):
                tasks.setdefault(table_name, []).append(
                    (
                        table_pair,
                        k,
                        RN,
                        num_folds,
                        "kfold",
                        distance_metric,
                        n_component,
                        target,
                    )
                )
    conn.close()
    return tasks


def process_table(args_list, results, table_path, table_name):
    # Load base embeddings (legacy blob table)
    ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table_path)
    targets = {
        "hit_freq": hfs,
        "mean_iou": mious,
        "mean_conf": mconfs,
        "flag_cat": cats,
        "flag_supercat": supercats,
    }
    distances = load_distances(table_path.replace("embeddings", "distances"))
    kf = KFold(n_splits=args_list[0][3], shuffle=True, random_state=args_list[0][2])

    new_args_list = []
    for args in args_list:
        new_args = list(args)
        if "_" in args[0]:
            # target from other model's embeddings
            root = os.path.dirname(table_path)
            other_model = args[0].split('.')[0].split('_')[1]
            other_targ = load_embeddings(os.path.join(root, f"{other_model}.db"), args[-1])[args[-1]].values
            new_args = [other_targ] + new_args + [kf]
        else:
            new_args = [targets[args[-1]]] + new_args + [kf]

        if "." in args[0]:
            # projection-based task: load vectors from proj DB using utils_vec
            # find proj DB path corresponding to args[0]
            # args[0] is model.algo.hash; determine algo and lookup proj DB
            algo = args[0].split('.')[1]
            proj_db = os.path.join(PROJ_DB_DIR, f"{algo}.db")
            embeddings = load_projection_vectors(proj_db, query=args[0])
            new_args = [embeddings] + new_args
        else:
            new_args = [distances] + new_args

        new_args_list.append(new_args)

    for args in new_args_list:
        results.extend(work_fn(args))

    return results


def KNN_vec(args):
    db_path = EMBEDDINGS_DIR
    tables = [os.path.join(db_path, file) for file in os.listdir(db_path) if file.endswith(".db")]
    logger = get_logger(args.debug)
    if not args.debug:
        import wandb

        run = wandb.init(entity="miro-unet", project="VLLM clustering", name="visu")
    else:
        run = None

    init_db()

    neighbor_grid = [5, 10, 15, 20, 30, 50]
    RN = 42
    num_folds = 10
    distance_metric = "euclidean"
    tasks = get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric)

    if len(tasks) > 0:
        records = []
        for table in tables:
            table_name = table.split("/")[-1].split(".")[0]
            if table_name not in tasks.keys():
                continue
            logger.info(f"Computing {len(tasks[table_name])} rows for table {table_name}")
            process_table(tasks[table_name], records, table, table_name)

        records = pd.DataFrame(data=[record.values() for record in records], columns=records[0].keys())
        if run is not None:
            import wandb

            run.log({"new_results": wandb.Table(data=records.values.tolist(), columns=list(records.columns))})

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM knn_results ORDER BY model", conn)
    conn.close()
    fig = plot_results(df)
    if run is not None:
        run.log({"plot": fig})
