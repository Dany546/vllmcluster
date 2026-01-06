import itertools
import os
import sqlite3
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
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
            distance_metric TEXT,
            fold INTEGER,
            target TEXT,
            mae REAL,
            r2 REAL,
            corr REAL,
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
        # Add column if missing
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


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

    query = (
        "model=? AND k=? AND random_state=? AND num_folds=? AND cv_type=? AND distance_metric=?"
        + query
    )
    keys = (model, k, random_state, num_folds, cv_type, distance_metric) + tuple(
        other_columns.values()
    )
    c.execute(
        f"""
        SELECT 1 FROM knn_results
        WHERE {query}
        LIMIT 1
        """,
        keys,
    )
    exists = c.fetchone()
    return exists is not None


def insert_record(record, c):
    for attempt in range(10):
        try:
            query = tuple(record.keys())
            placeholders = ", ".join(["?"] * len(query))
            values = tuple(record.values())
            c.execute(
                f"""
                INSERT OR IGNORE INTO knn_results ({", ".join(query)})
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


def get_lower_dim_tables():
    db_path = f"/globalscratch/ucl/irec/darimez/dino/proj/"
    lower_dim_tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if (
            file.endswith(".db")
            and file.split("/")[-1].split(".")[0] not in ["metrics"]
        )
    ]
    for db in ["tsne.db", "umap.db"]:
        conn = sqlite3.connect(os.path.join(db_path, db))
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
            f"{run_id.split('_')[0]}.{lower_table_name}.{run_id.split('_')[-1]}"
            for run_id in embed_runs
        ]
        tables.extend([lower_table] * len(new_tables))
        table_names.extend(new_tables)
        conn.close()

    return table_names, tables


def get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    targets = ["hit_freq", "mean_iou", "flag_supercat"]
    n_components = [2, 10, 50]
    k_targ = list(itertools.product(neighbor_grid, targets, n_components))
    tasks = {}  # list of (table_name, k)
    lower_dim_names, lower_dim_tables = get_lower_dim_tables()
    table_names = [
        table.split("/")[-1].split(".")[0] for table in tables
    ] + lower_dim_names
    tables = tables + lower_dim_tables
    for table, table_name in zip(tables, table_names):
        if ("clip" in table_name) or ("dino" in table_name):
            # Pair CLIP/DINO models with all non-CLIP/DINO models
            table_pairs = [
                f"{table_name}_{tn}"
                for tn in table_names
                if ("clip" not in tn) and ("dino" not in tn)
            ]
        else:
            # YOLO models remain singletons
            table_pairs = [table_name]

        k_targ_per_table = itertools.product(k_targ, table_pairs)
        for (k, target, n_component), table_pair in k_targ_per_table:
            other_columns = {"target": target, "n_components": n_component}
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
                        n_component,
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
        n_components,
        target_name,
        folds,
    ) = args

    # folds are now pre-generated and passed directly
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    records = []
    precomputed = distances.shape[0] == distances.shape[1]

    categorical_target = target_name in ["flag_cat", "flag_supercat"]
    if categorical_target:
        target = decode(target)
    KNN_class = KNeighborsClassifier if categorical_target else KNeighborsRegressor
    try:
        for fold_id, (tr_idx, te_idx) in enumerate(folds):
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

            record = {
                "model": model_name,
                "k": k,
                "cv_type": cv_type,
                "num_folds": len(folds),
                "random_state": random_state,
                "distance_metric": distance_metric,
                "fold": fold_id,
                "target": target_name,
                "mae": accuracy_score(y_test, y_pred)
                if categorical_target
                else mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "corr": adjusted_rand_score(y_test, y_pred)
                if categorical_target
                else np.corrcoef(y_test, y_pred)[0, 1],
            }
            records.append(record)
            success = insert_record(record, c)

        conn.commit()
    except Exception as e:
        print(f"Error processing table {model_name}: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()
    return records


def process_table(args_list, results, table_path, table_name):
    ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table_path)
    targets = {
        "hit_freq": hfs,
        "mean_iou": mious,
        "mean_conf": mconfs,
        "flag_cat": cats,
        "flag_supercat": supercats,
    }
    distances = load_distances(table_path.replace("embeddings", "distances"))
    # Use pre-generated deterministic fold indices for consistency
    folds = get_fold_indices(n_splits=args_list[0][3], random_state=42)

    new_args_list = []
    for args in args_list:
        new_args = list(args)
        if "_" in args[0]:  # need target error from other models
            root = os.path.dirname(table_path)
            other_targ = load_embeddings(
                os.path.join(root, f"{args[0].split('.')[0].split('_')[1]}.db"),
                args[-1],
            )[args[-1]].values
            new_args = [other_targ] + new_args + [folds]
        else:
            new_args = [targets[args[-1]]] + new_args + [folds]
        if "." in args[0]:  # need embeddings from projections
            path = table_path.replace("embeddings", "proj")
            path = "/".join(path.split("/")[:-1])
            path = path + f"/{args[0].split('.')[1]}.db"
            embeddings = load_embeddings(
                path,
                query=args[0],
            )
            new_args = [embeddings] + new_args
        else:
            new_args = [distances] + new_args
        new_args_list.append(new_args)
    for args in new_args_list:
        results.extend(work_fn(args))

    return results


import plotly.express as px
import plotly.graph_objects as go


def plot_results(df):
    df["model"] = df.assign(
        merged=df[["model", "distance_metric"]].astype(str).agg("_".join, axis=1)
    )["merged"]
    df.drop(columns=["distance_metric"], inplace=True)

    grid = {
        "target": ["hit_freq", "mean_iou", "mean_conf", "cats", "supercats"],
        "target_name": [
            "Hit Frequency",
            "Mean IoU",
            "Mean Confidence",
            "Categories",
            "Super Categories",
        ],
        "metrics": ["corr", "mae", "r2"],
        "metrics_name": ["Correlation", "Mean Absolute Error", "R-squared"],
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
            fig.add_trace(
                go.Box(
                    x=sub["k"],
                    y=sub[metric],
                    name=f"{model}",
                    marker_color=colors[i % len(colors)],
                    boxmean=True,
                    visible=(metric == "corr" and target == "mean_iou"),
                )
            )

    # Metric ranges
    metric_ranges = {
        "corr": [-1, 1],
        "mae": [0, df["mae"].max() + 0.05],
        "r2": [df["r2"].min() - 0.05, 1],
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
    db_path = f"/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db")
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
    # Use fixed random seed for determinism across all hyperparameter combinations
    RN = 42
    num_folds = 10
    distance_metric = "euclidean"
    tasks = get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric)

    if len(tasks) > 0:
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

        records = pd.DataFrame(
            data=[record.values() for record in records], columns=records[0].keys()
        )
        run.log(
            {
                "new_results": wandb.Table(
                    data=records.values.tolist(),
                    columns=list(records.columns),
                )
            }
        )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM knn_results ORDER BY model", conn)
    conn.close()
    fig = plot_results(df)
    run.log({"plot": fig})
