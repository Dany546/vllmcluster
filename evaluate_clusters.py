import os
import sqlite3
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from utils import get_logger, load_distances, load_embeddings

import wandb

DB_PATH = "/globalscratch/ucl/irec/darimez/dino/knn_results.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- Recommended PRAGMAs for multiprocessing ---
    c.execute("PRAGMA journal_mode=WAL;")  # Enables concurrent reads/writes
    c.execute("PRAGMA synchronous=NORMAL;")  # Faster commits, safe with WAL
    c.execute("PRAGMA temp_store=MEMORY;")  # Faster temp operations
    c.execute("PRAGMA busy_timeout=5000;")  # Wait up to 5s if DB is locked

    # Optional tuning (safe to keep, but not essential)
    c.execute("PRAGMA locking_mode=NORMAL;")  # Default; ensures shared access
    c.execute("PRAGMA cache_size=-20000;")  # ~20MB page cache (optional)

    conn.commit()

    # --- Create table ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS knn_results (
            model TEXT,
            k INTEGER,
            random_state INTEGER,
            num_folds INTEGER,
            cv_type TEXT,
            fold INTEGER,
            mse REAL,
            r2 REAL,
            corr REAL,
            PRIMARY KEY(model, k, random_state, num_folds, cv_type, fold)
        )
    """)

    conn.commit()
    conn.close()


def already_computed(model, k, random_state, num_folds, cv_type, c):
    c.execute(
        """
        SELECT 1 FROM knn_results
        WHERE model=? AND k=? AND random_state=? AND num_folds=? AND cv_type=?
        LIMIT 1
    """,
        (model, k, random_state, num_folds, cv_type),
    )
    exists = c.fetchone() is not None
    return exists


def insert_record(record, c):
    for attempt in range(10):
        try:
            c.execute(
                """
                INSERT OR IGNORE INTO knn_results (model, k, random_state, num_folds, cv_type, fold, mse, r2, corr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record["model"],
                    record["k"],
                    record["random_state"],
                    record["num_folds"],
                    record["cv_type"],
                    record["fold"],
                    record["mse"],
                    record["r2"],
                    record["corr"],
                ),
            )
            break
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.1 * (attempt + 1))
            else:
                raise
    raise RuntimeError("DB remained locked after retries")


def get_tasks(tables, neighbor_grid, RN, num_folds):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    tasks = {}  # list of (table_name, k)
    for table in tables:
        table_name = table.split("/")[-1].split(".")[0]
        for k in neighbor_grid:
            if not already_computed(table_name, k, RN, num_folds, "kfold", c):
                if table_name not in tasks:
                    tasks[table_name] = []
                tasks[table_name].append((table_name, k, RN, num_folds, "kfold"))
    conn.close()
    return tasks


def work_fn(args):
    (
        distances,
        mious,
        model_name,
        k,
        random_state,
        num_folds,
        cv_type,
    ) = args

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    indices = np.arange(5000)
    folds = list(kf.split(indices))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    records = []
    try:
        for fold_id, (tr_idx, te_idx) in enumerate(folds):
            # compute KNN fold
            D_train = distances[tr_idx][:, tr_idx]
            D_test = distances[te_idx][:, tr_idx]

            y_train = mious[tr_idx]
            y_test = mious[te_idx]

            knn = KNeighborsRegressor(
                n_neighbors=k,
                metric="precomputed",
                weights="distance",
            )

            knn.fit(D_train, y_train)
            y_pred = knn.predict(D_test)

            record = {
                "model": model_name,
                "k": k,
                "cv_type": cv_type,
                "num_folds": len(folds),
                "random_state": random_state,
                "fold": fold_id,
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "corr": np.corrcoef(y_test, y_pred)[0, 1],
            }
            records.append(record)
            insert_record(record, c)
            insert_record(record, c)

        conn.commit()
    except Exception as e:
        print(f"Error processing table {table_name}: {e}")
        conn.rollback()
    finally:
        conn.close()
        return records


def process_table(args_list, results, table_path, table_name):
    ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table_path)
    distances = load_distances(table_path.replace("embeddings", "distances"))

    args_list = [(distances, mious) + args for args in args_list]
    with Pool(processes=4) as pool:
        results_per_k = pool.map(work_fn, args_list)

    for recs in results_per_k:
        results.extend(recs)


# def perform_kfold(
#     distances,
#     mious,
#     folds,
#     records,
#     model_name,
#     logger,
#     neighbor_grid=[5, 10, 15, 20, 30, 50],
#     RN=0,
# ):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     try:
#         for k in neighbor_grid:
#             if already_computed(model_name, k, RN, len(folds), "kfold", c):
#                 logger.info(
#                     f"Skipping already computed k={k}, model={model_name}, random_state={RN}, num_folds={len(folds)}, cv_type='kfold'"
#                 )
#                 continue
#             for fold_id, (tr_idx, te_idx) in enumerate(folds):
#                 D_train = distances[tr_idx][:, tr_idx]
#                 D_test = distances[te_idx][:, tr_idx]

#                 y_train = mious[tr_idx]
#                 y_test = mious[te_idx]

#                 knn = KNeighborsRegressor(
#                     n_neighbors=k,
#                     metric="precomputed",
#                     weights="distance",
#                 )

#                 knn.fit(D_train, y_train)
#                 y_pred = knn.predict(D_test)

#                 record = {
#                     "model": model_name,
#                     "k": k,
#                     "cv_type": "kfold",
#                     "num_folds": len(folds),
#                     "random_state": RN,
#                     "fold": fold_id,
#                     "mse": mean_squared_error(y_test, y_pred),
#                     "r2": r2_score(y_test, y_pred),
#                     "corr": np.corrcoef(y_test, y_pred)[0, 1],
#                 }
#                 records.append(record)
#                 insert_record(record, c)
#             conn.commit()
#     except Exception as e:
#         logger.error(f"Error occurred during evaluation: {e}")
#     finally:
#         conn.close()
#     return records


def KNN(args):
    db_path = f"/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db")
        and table.split("/")[-1].split(".")[0] not in ["metrics", "umap", "tsne"]
    ]  # extend as needed
    logger = get_logger(args.debug)
    if not args.debug:
        run = wandb.init(
            entity="miro-unet",
            project="VLLM clustering",
            # mode="offline",
            name=f"visu",  # optional descriptive name
        )
    else:
        run = wandb

    init_db()

    neighbor_grid = [5, 10, 15, 20, 30, 50]
    n_samples = 5000
    RN = 42
    num_folds = 10
    tasks = get_tasks(tables, neighbor_grid, RN, num_folds)

    records = []

    for table in tables:
        # Start a new wandb run per model
        table_name = table.split("/")[-1].split(".")[0]
        if table_name not in tasks.keys():
            continue
        else:
            logger.info(
                f"Computing {len(tasks[table_name])} rows for table {table_name}"
            )
        process_table(tasks[table_name], records, table, table_name)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM knn_results", conn)
    conn.close()
    run.log(
        {
            "table": wandb.Table(
                data=df.values.tolist(),
                columns=list(df.columns),
            )
        }
    )
