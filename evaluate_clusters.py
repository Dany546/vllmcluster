import itertools
import os
import sqlite3
from joblib import Parallel, delayed
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from utils import get_logger, load_distances, load_embeddings

import wandb
from tqdm import tqdm

DB_PATH = "/CECI/home/ucl/irec/darimez/knn_results.db"


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
            distance_metric TEXT,
            fold INTEGER,
            target TEXT,
            "mae/accuracy" REAL,
            r2 REAL,
            "correlation/ARI" REAL,
            "error_dist_corr" REAL,
            PRIMARY KEY(model, k, random_state, num_folds, cv_type, distance_metric, fold, target)
        )
    """)

    conn.commit()
    conn.close()


def ensure_column_exists(c, table_name, column_name, column_type="TEXT"):
    # Check schema
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in c.fetchall()]  # row[1] is column name
    if column_name not in columns:
        # Add column if missing (quote names to support special characters)
        c.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type}')


def already_computed(
    model,
    k,
    random_state,
    num_folds,
    cv_type,
    distance_metric,
    other_columns=None,
    c: sqlite3.Cursor = None,
):
    if other_columns is None:
        other_columns = {}
        query = ""
    else:
        query = " AND " + "=? AND ".join(other_columns.keys()) + "=?"
    for col in other_columns.keys():
        # other_columns maps column_name -> value; ensure the column exists
        # but do not treat the value as an SQL type (would produce syntax errors).
        ensure_column_exists(c, "knn_results", col)

    # Only treat a run as "already computed" if we have at least `num_folds`
    # rows for the given parameter combination (i.e., all folds were saved).
    query = (
        "model=? AND k=? AND random_state=? AND num_folds=? AND cv_type=? AND distance_metric=?"
        + query
    )
    keys = (model, k, random_state, num_folds, cv_type, distance_metric) + tuple(
        other_columns.values()
    )
    c.execute(
        f"SELECT COUNT(*) FROM knn_results WHERE {query}",
        keys,
    )
    fetch_res = c.fetchone()
    cnt = fetch_res[0] if fetch_res is not None else None
    # If COUNT(*) returned None for some reason, fall back to previous behaviour
    if cnt is None:
        c.execute(
            f"SELECT 1 FROM knn_results WHERE {query} LIMIT 1",
            keys,
        )
        exists = c.fetchone()
        return exists is not None

    try:
        return int(cnt) >= int(num_folds)
    except Exception:
        # On any unexpected issue, be conservative and assume not computed
        return False


def insert_record(record, c):
    for attempt in range(10):
        try:
            # Compatibility mapping: if legacy keys are used in `record` but the
            # current table schema uses slightly different column names, map
            # them to an existing column to avoid OperationalError.
            c.execute("PRAGMA table_info(knn_results)")
            table_cols = [row[1] for row in c.fetchall()]  

            query = tuple(record.keys())
            placeholders = ", ".join(["?"] * len(query))
            values = tuple(record.values())
            columns_sql = ", ".join([f'"{col}"' for col in query])
            c.execute(
                f"""
                INSERT OR IGNORE INTO knn_results ({columns_sql})
                VALUES ({placeholders})
            """,
                values,
            )
            return True
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.1 * (attempt + 1))
            else:
                raise
    raise RuntimeError("DB remained locked after retries")


def get_lower_dim_tables(model_filter=None):
    db_path = f"/CECI/home/ucl/irec/darimez/proj/"
    lower_dim_tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if (
            file.endswith(".db")
            and file.split("/")[-1].split(".")[0] not in ["metrics"]
            and not ("attention.db" == file.split("/")[-1])
        )
    ]
    model_filter = [os.path.basename(mf).replace(".db", "") for mf in model_filter] if model_filter is not None else None
    for db in ["tsne.db", "umap.db"]:
        dbfile = os.path.join(db_path, db)
        if not os.path.exists(dbfile):
            continue
        try:
            conn = sqlite3.connect(dbfile)
            c = conn.cursor()
            # Try to remove orphan metadata entries. Support both legacy `embeddings`
            # table and sql-vector `vec_projections` fallback.
            try:
                c.execute(
                    """ DELETE FROM metadata WHERE run_id NOT IN ( SELECT DISTINCT run_id FROM embeddings ); """
                )
            except sqlite3.OperationalError:
                try:
                    c.execute(
                        """ DELETE FROM metadata WHERE run_id NOT IN ( SELECT DISTINCT run_id FROM vec_projections ); """
                    )
                except Exception:
                    # nothing we can do safely here
                    pass
            conn.commit()
        except Exception:
            # If the lower-dim DB cannot be opened, continue gracefully
            continue
    # Load embedding runs
    tables = []
    table_names = []
    for lower_table in lower_dim_tables:
        lower_table_name = lower_table.split("/")[-1].split(".")[0]
        conn = sqlite3.connect(lower_table)
        c = conn.cursor()
        # Prefer legacy `embeddings` table, but fall back to vec_projections when present
        try:
            c.execute("SELECT DISTINCT run_id FROM embeddings")
            embed_runs = {row[0] for row in c.fetchall()}
        except sqlite3.OperationalError:
            try:
                c.execute("SELECT DISTINCT run_id FROM vec_projections")
                embed_runs = {row[0] for row in c.fetchall()}
            except sqlite3.OperationalError:
                embed_runs = set()
        new_tables = [
            run_id for run_id in embed_runs 
            if (model_filter is None or any(mf in run_id for mf in model_filter))
        ]
        new_tables = [
            f"{run_id.split('_')[0]}.{lower_table_name}.{run_id.split('_')[-1]}"
            for run_id in new_tables 
        ]
        new_table_names = [
            nt.split('.')[0]
            for nt in new_tables 
        ]
        tables.extend(new_tables)
        table_names.extend(new_table_names)
        conn.close()

    return tables, table_names


def get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric, model_filter=None, target_filter=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Default base targets: mean_iou and classification loss. seg_loss
    # will be added only for segmentation runs (table names containing '-seg'). 
    base_targets = ["hit_freq", "mean_iou", "flag_supercat", "box_loss", "cls_loss", "dfl_loss"]

    tasks = {}  # list of (table_name, k)
    lower_dim_tables, lower_dim_names = [], [] # get_lower_dim_tables(model_filter=tables)
    table_names = [table.split("/")[-1].split(".")[0] for table in tables]
    all_tables_names = table_names + lower_dim_names
    all_tables = table_names + lower_dim_tables
    for table, table_name in zip(all_tables, all_tables_names):
        if ("clip" in table) or ("dino" in table):
            # Pair CLIP/DINO models with all non-CLIP/DINO models
            # skip any 'attention' projection/table name
            table_pairs = [
                f"{table}_{tn}"
                for tn in table_names
                if (("clip" not in tn) 
                    and ("dino" not in tn) 
                    and (tn != "attention") 
                    and ("umap" not in tn) 
                    and ("tsne" not in tn))
            ]
        else:
            # YOLO models remain singletons
            table_pairs = [table]
        for table_pair in table_pairs:
            # If a model_filter is provided, only include table_pairs that match
            if model_filter is not None and len(model_filter) > 0:
                matches = any(mf in table_pair for mf in model_filter)
                if not matches:
                    continue
            # Determine targets for this specific table_pair. Add seg_loss only when
            # the run name indicates segmentation (e.g., contains '-seg'). If a
            # `target_filter` is provided, intersect with it to allow explicit selection.
            targets_for_pair = base_targets.copy()
            if "-seg" in table_pair:
                targets_for_pair.append("seg_loss")
            # If caller provided an explicit target list, restrict to it
            if target_filter is not None and len(target_filter) > 0:
                targets_for_pair = [t for t in targets_for_pair if t in target_filter]
            
            for k, target in itertools.product(neighbor_grid, targets_for_pair):
                other_columns = {"target": target}
                if not already_computed(
                    table_pair, k, RN, num_folds, "kfold", distance_metric, other_columns, c
                ):
                    tasks.setdefault(table_name, []).append(
                        (
                            table_pair,
                            k,
                            RN,
                            num_folds,
                            "kfold",
                            distance_metric,
                            target, # must always be last arg
                        )
                    )
    conn.close()
    return tasks


def decode(x):
    for ix, xx in enumerate(x):
        if isinstance(xx, bytes):
            x[ix] = int.from_bytes(xx, "little")
        else:
            x[ix] = int(xx)
    return np.array(x, dtype=int).ravel()


def get_fold_indices(n_splits=10, random_state=42):
    """
    Generate and return deterministic fold indices for consistent cross-validation.
    This ensures the same data splits are used across all runs.
    """
    indices = np.arange(5000)  # Fixed dataset size
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(indices))


def work_fn(args):
    (
        distances,
        target,
        model_name,
        k,
        random_state,
        num_folds,
        cv_type,
        distance_metric,
        target_name,
        folds,
    ) = args

    # folds are now pre-generated and passed directly
    records = []
    precomputed = distances.shape[0] == distances.shape[1]

    categorical_target = target_name in ["flag_cat", "flag_supercat"]
    if categorical_target:
        target = decode(target)
    logger = get_logger(True)
    KNN_class = KNeighborsClassifier if categorical_target else KNeighborsRegressor

    def _process_fold(fold_id, tr_idx, te_idx):
        try:
            # compute KNN fold
            if precomputed:
                D_train = distances[tr_idx][:, tr_idx]
                D_test = distances[te_idx][:, tr_idx]
            else:
                D_train = distances[tr_idx]
                D_test = distances[te_idx]

            y_train = target[tr_idx]
            y_test = target[te_idx]

            distance_method = "uniform" if "uniform" in distance_metric else "distance"
            knn = KNN_class(
                n_neighbors=k,
                metric="precomputed" if precomputed else "minkowski",
                weights=distance_method,
            )

            knn.fit(D_train, y_train)
            y_pred = knn.predict(D_test)

            try:
                dists, neigh_idx = knn.kneighbors(D_test, n_neighbors=k, return_distance=True)
            except TypeError:
                dists, neigh_idx = knn.kneighbors(D_test, n_neighbors=k)
            avg_dists = np.mean(dists, axis=1)

            if categorical_target:
                sample_error = (y_pred != y_test).astype(int)
            else:
                sample_error = np.abs(y_pred - y_test)
            try:
                se_const = np.all(sample_error == sample_error[0]) if sample_error.size > 0 else True
            except Exception:
                se_const = False
            try:
                ad_const = np.all(avg_dists == avg_dists[0]) if avg_dists.size > 0 else True
            except Exception:
                ad_const = False

            if se_const or ad_const or np.isnan(avg_dists).any() or np.isinf(avg_dists).any():
                print(
                    "Constant/invalid input detected for spearmanr:\n",
                    f"model: {model_name}",
                    f"\nconstant err: {se_const}",
                    f"\nconstant dist: {ad_const}",
                    f"\nk: {k}",
                    f"\nfold: {fold_id}",
                    f"\ntarget: {target_name}",
                    f"\nsample_error_shape: {getattr(sample_error, 'shape', None)}",
                    f"\nsample_error_unique: {np.unique(sample_error).tolist() if sample_error.size<=50 else f'{np.unique(sample_error).size} unique'}",
                    f"\navg_dists_shape: {getattr(avg_dists, 'shape', None)}",
                    f"\navg_dists_stats: ",
                        "\n    min: " + str(float(np.nanmin(avg_dists)) if avg_dists.size>0 and not np.isnan(avg_dists).all() else None),
                        "\n    max: " + str(float(np.nanmax(avg_dists)) if avg_dists.size>0 and not np.isnan(avg_dists).all() else None),
                        "\n    mean: " + str(float(np.nanmean(avg_dists)) if avg_dists.size>0 and not np.isnan(avg_dists).all() else None),
                        "\n    nan_any: " + str(bool(np.isnan(avg_dists).any())),
                        "\n    inf_any: " + str(bool(np.isinf(avg_dists).any())), 
                )
                # Removed blocking input() to avoid EOFError in worker processes
                # when running under joblib/loky. Log a brief warning instead.
                logger.warning("Constant/invalid input detected for spearmanr; continuing without interactive pause")

            err_dist_corr = spearmanr(sample_error, avg_dists)[0] 

            record = {
                "model": model_name,
                "k": k,
                "cv_type": cv_type,
                "num_folds": len(folds),
                "random_state": random_state,
                "distance_metric": distance_metric,
                "fold": fold_id,
                "target": target_name,
                "mae/accuracy": accuracy_score(y_test, y_pred)
                if categorical_target
                else mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "correlation/ARI": adjusted_rand_score(y_test, y_pred)
                if categorical_target
                else spearmanr(y_test, y_pred)[0],
                "error_dist_corr": float(err_dist_corr) if err_dist_corr is not None and not np.isnan(err_dist_corr) else None,
            }
            return record
        except Exception as e:
            logger.error("Error processing fold %s for model %s: %s", fold_id, model_name, e)
            raise
            
    try:
        # Use joblib to parallelize fold processing
        results = Parallel(n_jobs=-1, backend="loky", verbose=1000)(
            delayed(_process_fold)(fold_id, tr_idx, te_idx)
            for fold_id, (tr_idx, te_idx) in enumerate(folds)
        )
        records.extend(results)
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            # --- Recommended PRAGMAs for multiprocessing ---
            c.execute("PRAGMA journal_mode=WAL;")  # Enables concurrent reads/writes
            c.execute("PRAGMA synchronous=NORMAL;")  # Faster commits, safe with WAL
            c.execute("PRAGMA temp_store=MEMORY;")  # Faster temp operations
            c.execute("PRAGMA busy_timeout=5000;")  # Wait up to 5s if DB is locked
            for record in records:
                insert_record(record, c) 
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Error inserting records for model %s: %s", model_name, e)
            if conn:
                conn.close()
            raise  
    except Exception as e:
        logger.error("Error processing table %s: %s", model_name, e)
        raise

    return records


def is_valid_target_array(targ_arr, n_samples, source_desc, target_name, logger):
    """Normalize and validate a loaded target array.

    Returns a 1D numpy array on success, or None if the target should be skipped.
    """
    try:
        targ = np.asarray(targ_arr).squeeze()
    except Exception:
        logger.warning("Skipping task: could not convert target '%s' from %s to array", target_name, source_desc)
        return None

    if getattr(targ, "size", 0) == 0:
        logger.warning("Skipping task: target '%s' not found or empty in %s", target_name, source_desc)
        return None

    if targ.size != n_samples:
        logger.warning(
            "Skipping task: target '%s' length (%d) != n_samples (%d) in %s",
            target_name,
            int(targ.size),
            int(n_samples),
            source_desc,
        )
        return None

    # ignore NaNs for uniqueness test
    try:
        unique_vals = np.unique(targ[~np.isnan(targ)])
    except Exception:
        unique_vals = np.unique(targ)

    # treat single-unique-value or all-zero as invalid for our experiments
    if unique_vals.size <= 1 or np.all(targ == 0):
        logger.warning(
            "Skipping task: target '%s' is constant or all-zero in %s (unique=%d)",
            target_name,
            source_desc,
            int(unique_vals.size),
        )
        return None

    return targ


def process_table(args_list, results, table_path, table_name, pbar=None, n_jobs=-1, batch_size=32):
    logger = get_logger(True)
    # Try to load precomputed distances (preferred); fall back to embeddings matrix
    distances = None
    try:
        distances = load_distances(table_path)
    except Exception:
        # Try alternative conventional locations
        try:
            distances = load_distances(table_path.replace("embeddings", "distances"))
        except Exception:
            distances = None

    # Load embeddings/metadata if available (may be in a parallel 'embeddings' DB)
    ids = X = hfs = mious = mconfs = cats = supercats = losses = None
    try:
        emb_path = table_path.replace("distances", "embeddings")
        _emb_res = load_embeddings(emb_path)
    except Exception:
        try:
            _emb_res = load_embeddings(table_path)
        except Exception:
            _emb_res = None

    if _emb_res is not None:
        try:
            ids, X, hfs, mious, mconfs, cats, supercats, losses = _emb_res
        except Exception:
            # If load_embeddings returned a raw embedding matrix for lower-dim tables
            X = _emb_res
            ids = None

    targets = {
        "hit_freq": hfs,
        "mean_iou": mious,
        "mean_conf": mconfs,
        "flag_cat": cats,
        "flag_supercat": supercats,
        # losses may be None or arrays inside the returned dict
        "box_loss": losses.get("box_loss") if isinstance(losses, dict) else None,
        "cls_loss": losses.get("cls_loss") if isinstance(losses, dict) else None,
        "dfl_loss": losses.get("dfl_loss") if isinstance(losses, dict) else None,
        "seg_loss": losses.get("seg_loss") if isinstance(losses, dict) else None,
    }
    # distances = X # 
    distances = load_distances(table_path.replace("embeddings", "distances"))
    # Use pre-generated deterministic fold indices for consistency
    folds = get_fold_indices(n_splits=args_list[0][3], random_state=42)

    # Determine number of samples from distances or embeddings
    n_samples = None
    if distances is not None:
        n_samples = distances.shape[0]
    elif X is not None:
        try:
            n_samples = X.shape[0]
        except Exception:
            n_samples = None
    if n_samples is None:
        raise RuntimeError(f"Could not determine number of samples for table {table_path}")

    # We'll construct task arguments on-the-fly and execute them in batches
    batch = []
    processed = 0
    logger = get_logger(True)
    for args in args_list:
        new_args = list(args)
        # handle targets coming from other model runs (args[0] like 'run.proj.id')
        if "_" in args[0]:
            root = os.path.dirname(table_path)
            other_db = os.path.join(root, f"{args[0].split('_')[1]}.db")
            try:
                other_df = load_embeddings(other_db, args[-1])
            except Exception:
                logger.warning("Skipping task: could not load target '%s' from %s", args[-1], other_db)
                continue
            try:
                other_targ = other_df[args[-1]].values
            except Exception:
                other_targ = np.asarray(other_df).squeeze()

            valid = is_valid_target_array(other_targ, distances.shape[0], other_db, args[-1], logger)
            if valid is None:
                continue
            new_args = [valid] + new_args + [folds]
        else:
            target_name = args[-1]
            # If a preloaded target exists and is not None, use it
            if target_name in targets and targets[target_name] is not None:
                targ_arr = targets[target_name]
            else:
                df_single = load_embeddings(table_path, query=target_name)
                if isinstance(df_single, pd.DataFrame) and target_name in df_single.columns:
                    targ_arr = df_single[target_name].values
                else:
                    try:
                        if hasattr(df_single, "squeeze"):
                            targ_arr = df_single.squeeze().values if hasattr(df_single.squeeze(), "values") else df_single.squeeze()
                        else:
                            targ_arr = np.asarray(df_single)
                    except Exception:
                        logger.warning("Skipping task: Target column '%s' not found in %s", target_name, table_path)
                        continue

            valid = is_valid_target_array(targ_arr, distances.shape[0], table_path, target_name, logger)
            if valid is None:
                continue
            new_args = [valid] + new_args + [folds]
        if "." in args[0]:  # need embeddings from projections
            # args[0] is expected to be like '<run>.<proj>.<id>' or similar; extract the projection name
            try:
                parts = args[0].split('.')
                proj_name = parts[1] if len(parts) > 1 else parts[0]
            except Exception:
                proj_name = args[0].replace('.', '_')

            # Build a proj DB path relative to the embeddings DB
            root = os.path.dirname(table_path)
            if "embeddings" in root:
                proj_dir = root.replace("embeddings", "proj")
            else:
                proj_dir = os.path.join(root, "proj")

            proj_db = os.path.join(proj_dir, f"{proj_name}.db")

            try:
                embeddings = load_embeddings(proj_db, query=args[0].split('_')[0])
            except Exception as e:
                raise e
            new_args = [embeddings] + new_args
        else:
            new_args = [distances] + new_args
        # append constructed task args to batch
        batch.append(new_args)

        # If batch is full, run it with joblib in parallel
        if len(batch) >= batch_size:
            try:
                parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                    delayed(work_fn)(b) for b in batch
                )
                # Each parallel result is a list of records (one per fold); extend results
                for pr in parallel_results:
                    results.extend(pr)
            except Exception as e:
                logger.error("Error running batch for table %s: %s", table_name, e)
                raise
            processed += len(batch)
            if pbar is not None:
                try:
                    pbar.update(len(batch))
                except Exception:
                    pass
            batch = []

    # Process remaining tasks in the final batch
    if len(batch) > 0:
        try:
            parallel_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(work_fn)(b) for b in batch
            )
            for pr in parallel_results:
                results.extend(pr)
        except Exception as e:
            logger.error("Error running final batch for table %s: %s", table_name, e)
            raise
        if pbar is not None:
            try:
                pbar.update(len(batch))
            except Exception:
                pass

    return results


import plotly.express as px
import plotly.graph_objects as go


def plot_results(df):
    # Work on a copy to avoid mutating caller data
    df = df.copy()

    # Normalize legacy metric column names produced by the KNN job
    # older records used 'correlation/ARI' and 'mae/accuracy'
    if "correlation/ARI" in df.columns and "corr" not in df.columns:
        df["corr"] = df["correlation/ARI"]
    if "mae/accuracy" in df.columns and "mae" not in df.columns:
        df["mae"] = df["mae/accuracy"]
    # ensure r2 exists
    if "r2" not in df.columns:
        df["r2"] = np.nan

    # If distance_metric exists, merge into model string as before
    if "distance_metric" in df.columns and "model" in df.columns:
        df["model"] = df.assign(
            merged=df[["model", "distance_metric"]].astype(str).agg("_".join, axis=1)
        )["merged"]
        df.drop(columns=["distance_metric"], inplace=True, errors="ignore")

    grid = {
        # use canonical target keys used elsewhere in the code
        "target": ["hit_freq", "mean_iou", "mean_conf", "box_loss", "cls_loss", "dfl_loss", "seg_loss", "flag_cat", "flag_supercat"],
        "target_name": [
            "Hit Frequency",
            "Mean IoU",
            "Mean Confidence",
            "Box Loss",
            "Classification Loss",
            "DFL Loss",
            "Segmentation Loss",
            "Categories",
            "Super Categories",
        ],
        "metrics": ["correlation/ARI", "mae/accuracy", "r2", "error_dist_corr"],
        "metrics_name": ["Correlation/Adjusted Rank Index", "Mean Absolute Error/ accuracy", "R-squared", "Error-Distance Correlation"],
    }

    colors = px.colors.qualitative.Pastel
    fig = go.Figure()

    models = [m for m in df["model"].unique() if m not in ("clip", "dino")]
    n_models = len(models)
    n_targets = len(grid["target"])
    n_metrics = len(grid["metrics"])

    # Add traces: metric × target × model
    for metric, target in itertools.product(grid["metrics"], grid["target"]):
        for i, model in enumerate(models):
            sub = df[(df["model"] == model) & (df["target"] == target)]
            # safe access to columns to avoid KeyError
            x = sub["k"] if "k" in sub.columns else pd.Series(dtype=float)
            y = sub[metric] if metric in sub.columns else pd.Series(dtype=float)
            fig.add_trace(
                go.Box(
                    x=x,
                    y=y,
                    name=f"{model}",
                    marker_color=colors[i % len(colors)],
                    boxmean=True,
                    visible=(metric == "corr" and target == "mean_iou"),
                )
            )

    # Metric ranges (guard against missing columns)
    metric_ranges = {
        "corr": [-1, 1],
        "mae": [0, float(df["mae"].max(skipna=True) if "mae" in df.columns and not df["mae"].isna().all() else 1.0)],
        "r2": [float(df["r2"].min(skipna=True) if "r2" in df.columns and not df["r2"].isna().all() else -1.0), 1],
        "error_dist_corr": [-1, 1],
    }

    # Default selections for coordination
    default_metric_idx = 0  # corr
    default_target_idx = 1  # mean_iou

    # Metric dropdown - each button shows that metric with ALL targets
    metric_buttons = []
    for m_idx, (metric, metric_name) in enumerate(
        zip(grid["metrics"], grid["metrics_name"])
    ):
        # Create sub-buttons for each target
        for t_idx, (target, target_name) in enumerate(
            zip(grid["target"], grid["target_name"])
        ):
            visibility = [
                (trace_idx // (n_targets * n_models)) == m_idx
                and ((trace_idx % (n_targets * n_models)) // n_models) == t_idx
                for trace_idx in range(n_metrics * n_targets * n_models)
            ]
            metric_buttons.append(
                dict(
                    label=f"{metric_name} - {target_name}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "yaxis.range": metric_ranges[metric],
                            "title": f"{metric_name} - {target_name}",
                        },
                    ],
                )
            )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=0.0,
                y=1.15,
                showactive=True,
                buttons=metric_buttons,
            ),
        ],
        title=f"{grid['metrics_name'][default_metric_idx]} - {grid['target_name'][default_target_idx]}",
        xaxis_title="k neighbors",
        yaxis_title="metric value",
        yaxis_range=metric_ranges[grid["metrics"][default_metric_idx]],
        template="plotly_white",
        boxmode="group",
    )
    return fig


def KNN(args):
    db_path = f"/CECI/home/ucl/irec/darimez/distances/"
    all_tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db") and not file=='attention.db'
    ]  # extend as needed
    logger = get_logger(args.debug)
    if True:
        # If user provided specific table(s) via CLI, filter down the table list.
        if getattr(args, "table", None):
            logger.debug(f"Filtering tables based on --table: {args.table}")  # DEBUG
            requested = [t.strip() for t in args.table.split(",") if t.strip()]
            tables = []
            for req in requested:
                # Allow absolute paths to .db files or basenames (with or without .db)
                if os.path.isabs(req) and req.endswith(".db") and os.path.exists(req):
                    tables.append(req)
                else:
                    bn = req.split(".")[0]
                    matches = [t for t in all_tables if os.path.basename(t).split(".")[0] == bn]
                    if matches:
                        tables.extend(matches)
                    else:
                        candidate = os.path.join(db_path, req if req.endswith(".db") else req + ".db")
                        if os.path.exists(candidate):
                            tables.append(candidate)
                        else:
                            logger.warning(f"Requested table {req} not found in {db_path}")
        else:
            tables = all_tables

        if len(tables) == 0:
            logger.info("No tables to process (filtered by --table). Exiting.")
            return

        if not args.debug:
            run = wandb.init(
                entity="miro-unet",
                project="VLLM clustering",
                # mode="offline",
                name=f"visu",  # optional descriptive name
            )
        else:
            class wandb_mockup:
                def log(self, *args, **kwargs):
                    pass 
                def finish(self):
                    pass
            run = wandb_mockup()

        init_db()

        neighbor_grid = [5, 10, 15, 20, 30, 50]
        # Use fixed random seed for determinism across all hyperparameter combinations
        RN = 42
        num_folds = 10
        distance_metric = "euclidean"
        # Allow narrowing the grid to a subset of models via CLI: --model_filter "yolov11x-seg,other"
        model_filter = None
        if getattr(args, "model_filter", None):
            model_filter = [m.strip() for m in args.model_filter.split(",") if m.strip()]
            logger.info(f"Applying model filter: {model_filter}")
        # Allow selecting which targets to run via CLI: --targets "mean_iou,cls_loss"
        target_filter = None
        if getattr(args, "targets", None):
            target_filter = [t.strip() for t in args.targets.split(",") if t.strip()]
            logger.info(f"Applying target filter: {target_filter}")

        # Debug: report discovered embedding DBs and table selection
        logger.debug(f"Found {len(all_tables)} embedding DB files in {db_path}")
        logger.debug(f"Tables after --table filter: {len(tables)}")
        logger.debug(f"Model filter: {model_filter}; Target filter: {target_filter}")

        tasks = get_tasks(
            tables,
            neighbor_grid,
            RN,
            num_folds,
            distance_metric,
            model_filter=model_filter,
            target_filter=target_filter,
        )
        total_tasks = sum(len(v) for v in tasks.values())
        logger.info(f"Total KNN tasks to process: {total_tasks}")
        
        # Debug: per-table task counts
        for tn, lst in tasks.items():
            logger.debug(f"Tasks scheduled for {tn}: {len(lst)}")
        if total_tasks > 0:
            records = []

            # Create a global progress bar across all scheduled tasks so we can
            # estimate remaining time. `tasks` maps table_name -> list of task tuples.
            total_tasks = sum(len(v) for v in tasks.values())
            pbar = tqdm(total=total_tasks, desc="KNN tasks", unit="task")

            for table in tables:
                # Start a new wandb run per model
                table_name = table.split("/")[-1].split(".")[0]
                if table_name not in tasks.keys():
                    continue
                else:
                    logger.info(
                        f"Computing {len(tasks[table_name])} rows for table {table_name}"
                    )
                process_table(tasks[table_name], records, table, table_name, pbar=pbar)

            try:
                pbar.close()
            except Exception:
                pass
            # If we processed tasks, upload results and plot to wandb
            if len(records) > 0:
                try:
                    records_df = pd.DataFrame(
                        data=[record.values() for record in records],
                        columns=list(records[0].keys()),
                    )
                    run.log(
                        {
                            "new_results": wandb.Table(
                                data=records_df.values.tolist(),
                                columns=list(records_df.columns),
                            )
                        }
                    )
                    fig = plot_results(records_df)
                    run.log({"plot": fig})
                except Exception as e:
                    logger.error("Failed to log results/plot to wandb: %s", e)
        else:
            # No tasks scheduled — provide diagnostic info to help the user debug
            logger.warning("No KNN tasks were scheduled.")
            if model_filter:
                # Check whether results already exist in DB for the requested filters
                conn_check = sqlite3.connect(DB_PATH)
                c_check = conn_check.cursor()
                for mf in model_filter:
                    c_check.execute(
                        "SELECT COUNT(*) FROM knn_results WHERE model LIKE ?", (f"%{mf}%",)
                    )
                    cnt = c_check.fetchone()[0]
                    logger.info(f"Existing results matching '{mf}': {cnt}")
                conn_check.close()
            logger.info("Possible reasons: filters matched no models, targets were filtered out, or results already exist in the DB.")
            # If no records were produced, verify whether this is because results
            # already exist in the DB. If not, raise so callers can debug the
            # unexpected skip (records should not be empty in normal execution).
            if len(records) == 0:
                expected_models = list(tasks.keys())
                if len(expected_models) > 0:
                    conn_check = sqlite3.connect(DB_PATH)
                    c_check = conn_check.cursor()
                    placeholders = ",".join(["?"] * len(expected_models))
                    c_check.execute(
                        f"SELECT COUNT(*) FROM knn_results WHERE model IN ({placeholders})",
                        expected_models,
                    )
                    cnt_existing = c_check.fetchone()[0]
                    conn_check.close()
                    if cnt_existing == 0:
                        raise RuntimeError(
                            "No records were produced for requested tasks and no existing results "
                            "found in knn_results. This indicates tasks were skipped unexpectedly."
                        )
            records = pd.DataFrame(
                data=[record.values() for record in records], columns=list(records[0].keys())
            )
            run.log(
                {
                    "new_results": wandb.Table(
                        data=records.values.tolist(),
                        columns=list(records.columns),
                    )
                }
            )
    else:
        for table in tables:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM knn_results ORDER BY model", conn)
            conn.close()
            fig = plot_results(df)
            run.log({"plot": fig})
