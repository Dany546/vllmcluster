import os
import sqlite3
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
            distance_metric TEXT,
            fold INTEGER,
            mae REAL,
            r2 REAL,
            corr REAL,
            PRIMARY KEY(model, k, random_state, num_folds, cv_type, distance_metric, fold)
        )
    """)

    conn.commit()
    conn.close()


def already_computed(model, k, random_state, num_folds, cv_type, distance_metric, c):
    c.execute(
        """
        SELECT 1 FROM knn_results
        WHERE model=? AND k=? AND random_state=? AND num_folds=? AND cv_type=? AND distance_metric=?
        LIMIT 1
    """,
        (model, k, random_state, num_folds, cv_type, distance_metric),
    )
    exists = c.fetchone()
    return exists is not None


def insert_record(record, c):
    for attempt in range(10):
        try:
            c.execute(
                """
                INSERT OR IGNORE INTO knn_results (model, k, random_state, num_folds, cv_type, distance_metric, fold, mae, r2, corr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record["model"],
                    record["k"],
                    record["random_state"],
                    record["num_folds"],
                    record["cv_type"],
                    record["distance_metric"],
                    record["fold"],
                    record["mae"],
                    record["r2"],
                    record["corr"],
                ),
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
    lower_dim_tables = []
    table_names = []
    for lower_table in lower_dim_tables:
        lower_table_name = lower_table.split("/")[-1].split(".")[0]
        conn = sqlite3.connect(lower_table)
        c = conn.cursor()
        c.execute(f"SELECT run_id, model FROM metadata ORDER BY model")
        new_tables = [
            f"{model}.{lower_table_name}.{run_id}" for run_id, model in c.fetchall()
        ]
        lower_dim_tables.extend([lower_table] * len(new_tables))
        table_names.extend(new_tables)
        conn.close()

    return table_names, lower_dim_tables


def get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    tasks = {}  # list of (table_name, k)
    lower_dim_names, lower_dim_tables = get_lower_dim_tables()
    table_names = [
        table.split("/")[-1].split(".")[0] for table in tables
    ] + lower_dim_names
    tables = tables + lower_dim_tables
    for table, table_name in zip(tables, table_names):
        table_pairs = (
            [table_name]
            if table_name not in ("clip", "dino")
            else [
                "_".join([table_name, tn])
                for tn in table_names
                if tn not in ("clip", "dino")
            ]
        )
        for k in neighbor_grid:
            for table_pair in table_pairs:
                if not already_computed(
                    table_pair, k, RN, num_folds, "kfold", distance_metric, c
                ):
                    if table_name not in tasks:
                        tasks[table_name] = []
                    tasks[table_name].append(
                        (table_pair, k, RN, num_folds, "kfold", distance_metric)
                    )
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
        distance_metric,
        kf,
    ) = args

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
                weights="unifrom" if "uniform" in distance_metric else "distance",
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
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "corr": np.corrcoef(y_test, y_pred)[0, 1],
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
    distances = load_distances(table_path.replace("embeddings", "distances"))
    kf = KFold(n_splits=args_list[0][3], shuffle=True, random_state=args_list[0][2])

    new_args_list = []
    for args in args_list:
        new_args = [args]
        if "_" in args[0]:  # need target error from other models
            root = os.path.dirname(table_path)
            other_mious = load_embeddings(
                os.path.join(root, f"{args[0].split('_')[1]}.db"),
                "mean_iou",
            )["mean_iou"].values
            new_args = [other_mious] + new_args + [kf]
        else:
            new_args = [mious] + new_args + [kf]
        if "." in args[0]:  # need target error from other models
            embeddings = load_embeddings(
                os.path.join(
                    table_path.replace("embeddings", "proj"), f"{args[0][1]}.db"
                ),
                query=args[0],
            )
            new_args = [embeddings] + new_args + [kf]
        else:
            new_args = [distances] + new_args + [kf]
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

    metrics = ["mae", "r2", "corr"]
    colors = px.colors.qualitative.Pastel

    fig = go.Figure()

    # Add one trace per metric per model
    for metric in metrics:
        models = df["model"].unique()
        models = [m for m in models if not m in ("clip", "dino")]
        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            fig.add_trace(
                go.Box(
                    x=sub["k"],
                    y=sub[metric],
                    name=f"{model} ({metric})",
                    marker_color=colors[i % len(colors)],
                    boxmean=True,
                    visible=(metric == "mae"),  # only MAE visible at start
                )
            )

    # Build dropdown menu
    buttons = []
    metric_ranges = {
        m: _range
        for m, _range in zip(
            metrics, [[0, df["mae"].max() + 0.05], [df["r2"].min() - 0.05, 1], [-1, 1]]
        )
    }
    n_models = df["model"].nunique()

    for m_idx, metric in enumerate(metrics):
        visibility = []
        for i in range(len(metrics)):
            for _ in range(n_models):
                visibility.append(i == m_idx)

        buttons.append(
            dict(
                label=metric.upper(),
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": f"{metric.upper()} distribution per k",
                        "yaxis": {"range": metric_ranges[metric]},
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=1.15,
                y=0.5,
                showactive=True,
                buttons=buttons,
            )
        ],
        title="MAE distribution per k",
        xaxis_title="k",
        yaxis_title="Metric value",
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
    RN = 42
    num_folds = 10
    distance_metric = "euclidean"
    tasks = get_tasks(tables, neighbor_grid, RN, num_folds, distance_metric)
    print("Tasks:", tasks)

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
