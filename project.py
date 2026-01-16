import hashlib
import itertools
import json
import logging
import os
import sqlite3
from collections import defaultdict

import concurrent

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from utils import dict_to_filename, get_logger, get_lookups, load_embeddings
from sqlvector_projector import (
    connect_vec_db,
    create_vec_tables,
    compute_and_store,
    compute_and_store_metrics,
    load_vectors_from_db,
    get_vector_slice,
    compute_run_id,
    compute_projection_to_file,
    store_projection_from_file,
)
from tqdm import tqdm   

EMBEDDINGS_DIR = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
PROJ_DB_DIR = "/CECI/home/ucl/irec/darimez" # "/globalscratch/ucl/irec/darimez/dino/proj/"
METRICS_DB = os.path.join(PROJ_DB_DIR, "metrics.db")

def _executemany_with_retry(cur, conn, query, rows, retries=8, base_delay=0.1):
    """Execute `executemany` with retries on transient sqlite locking/busy errors."""
    import time
    logger = logging.getLogger(__name__)
    for attempt in range(retries):
        try:
            cur.executemany(query, rows)
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "database is locked" in msg or "database is busy" in msg:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"SQLite busy/locked, retrying in {delay:.2f}s (attempt {attempt+1}/{retries}): {e}")
                time.sleep(delay)
                continue
            raise
    # Final attempt (let it raise if it fails)
    cur.executemany(query, rows)
    conn.commit()

# Global storage for worker initialization
_worker_X = None
_worker_model_name = None
_worker_args = None

def _init_worker(X, model_name, args):
    """Initializer function for worker processes - stores data in global to avoid pickle overhead"""
    global _worker_X, _worker_model_name, _worker_args
    _worker_X = X
    _worker_model_name = model_name
    _worker_args = args
    import multiprocessing as mp 
    import sys
    pid = os.getpid()
    print(f"[WORKER_INIT] PID={pid} started | CPU count={mp.cpu_count()}", flush=True, file=sys.stderr)

def compute_projection_task(task):
    """Helper function to compute a single projection task (computes embedding and writes it to a temp file).
    The main process will be responsible for storing the file into the projection DB to avoid concurrent writers."""
    import os
    import sys
    import gc
    
    pid = os.getpid()
    logger = get_logger(_worker_args.debug)
    
    algo = task["algo"]
    params = {k: v for k, v in task.items() if k != "algo"}

    try:
        # compute embedding and save to a temp file (worker-local) to avoid DB contention
        run_id = compute_run_id(_worker_model_name, params)
        tmp_dir = os.environ.get('SQLVEC_TMP', PROJ_DB_DIR)
        tmp_path = os.path.join(tmp_dir, f"proj_{algo}_{run_id}_{pid}.npy")
        compute_projection_to_file(_worker_X, _worker_model_name, params, out_path=tmp_path, algo=algo)
        # Clean up memory after each task
        gc.collect()
        return {"success": True, "task": task, "tmp_path": tmp_path, "run_id": run_id}
    except Exception as e:
        logger.error(f"Error computing projection for task {task}: {e}")
        return {"success": False, "task": task, "error": str(e)}


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
        projector = TSNE(**hyperparams, random_state=42, method="exact")
    else:
        projector = UMAP(**hyperparams, random_state=42)

    embedding = projector.fit_transform(X)

    # Connect to DB
    conn = sqlite3.connect(os.path.join(db_path, f"{algo}.db"))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER,
            run_id TEXT,
            vector BLOB,
            PRIMARY KEY(id, run_id),
            FOREIGN KEY(run_id) REFERENCES metadata(run_id)
        )
    """)
    
    # Insert embeddings in batches
    rows = []
    for i, vec in enumerate(embedding):
        # Serialize vector as bytes
        vec_blob = vec.astype(np.float32).tobytes()
        rows.append((int(i), run_id, vec_blob))
        
        if len(rows) >= 128:
            query = (
                "INSERT OR REPLACE INTO embeddings(id, run_id, vector) VALUES (?, ?, ?)"
            )
            _executemany_with_retry(cur, conn, query, rows)
            rows = []
            conn.commit()

    if len(rows) > 0:
        query = (
            "INSERT OR REPLACE INTO embeddings(id, run_id, vector) VALUES (?, ?, ?)"
        )
        _executemany_with_retry(cur, conn, query, rows)
        conn.commit()

    # Insert metadata with n_components
    cols = ", ".join(["model"] + list(hyperparams.keys()))
    placeholders = ", ".join(["?"] * (len(hyperparams) + 2))  # run_id + model + hyperparams
    sql = f"INSERT OR REPLACE INTO metadata(run_id, {cols}) VALUES ({placeholders})"
    cur.execute(sql, (run_id, model_name, *hyperparams.values()))

    conn.commit()
    conn.close()

    return run_id


def load_vectors_from_db(db_path, algo, run_id, ids=None):
    """
    Load vectors from database for a given run_id.
    
    Args:
        db_path: Path to database directory
        algo: Algorithm name ('tsne' or 'umap')
        run_id: Run identifier
        ids: Optional list of specific IDs to load (for visualization)
    
    Returns:
        numpy array of shape (n_samples, n_dims)
    """
    conn = sqlite3.connect(os.path.join(db_path, f"{algo}.db"))
    cur = conn.cursor()
    
    # Get n_components from metadata
    cur.execute("SELECT n_components FROM metadata WHERE run_id=?", (run_id,))
    result = cur.fetchone()
    if result is None:
        conn.close()
        raise ValueError(f"Run ID {run_id} not found in metadata")
    
    # Load vectors
    if ids is None:
        cur.execute("SELECT id, vector FROM embeddings WHERE run_id=? ORDER BY id", (run_id,))
    else:
        placeholders = ",".join("?" * len(ids))
        cur.execute(
            f"SELECT id, vector FROM embeddings WHERE run_id=? AND id IN ({placeholders}) ORDER BY id",
            (run_id, *ids)
        )
    
    rows = cur.fetchall()
    conn.close()
    
    # Deserialize vectors
    vectors = []
    for row_id, vec_blob in rows:
        vec = np.frombuffer(vec_blob, dtype=np.float32).reshape(-1)
        vectors.append(vec)
    
    return np.array(vectors)


def get_vector_slice(db_path, algo, run_id, ids=None, dims=[0, 1]):
    """
    Get specific dimensions from vectors (useful for 2D visualization).
    
    Args:
        db_path: Path to database directory
        algo: Algorithm name
        run_id: Run identifier
        ids: Optional specific IDs to load
        dims: List of dimension indices to extract (default [0,1] for x,y)
    
    Returns:
        numpy array of shape (n_samples, len(dims))
    """
    vectors = load_vectors_from_db(db_path, algo, run_id, ids)
    return vectors[:, dims]


def compute_and_store_metrics(model_name, data, db_path=""):
    anns_by_image, id_to_name, id_to_super, categories, supercategories = get_lookups()
    ids, hfs, mious, mconfs, cats, supercats = data

    conn = sqlite3.connect(os.path.join(db_path, f"metrics.db"))
    # Give SQLite a reasonable busy timeout to reduce transient lock errors
    conn.execute("PRAGMA busy_timeout = 5000")
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
            _executemany_with_retry(cur, conn, query, rows)
            rows = []
        rows.append(values)

    if len(rows) > 0:
        query = f"""INSERT OR IGNORE INTO metrics(model, img_id, hit_freq, mean_iou, mean_conf,
                            {cat_cols}, {super_cols}) VALUES ({placeholders})"""
        _executemany_with_retry(cur, conn, query, rows)
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
            os.path.join("/CECI/home/ucl/irec/darimez/proj", f"{algo}.db")
        )
        cur = conn.cursor()
        cols = ", ".join(
            ["model TEXT"] + [str(h) + " REAL" for h in hyperparam.keys() if h != "algo"]
        )
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS metadata (
                run_id TEXT PRIMARY KEY,
                {cols}
            )
        """)
        conn.commit()
        cur.execute("SELECT 1 FROM metadata WHERE run_id=?", (run_id,))
        exists = cur.fetchone() 
        if exists is None:
            tasks.append(hyperparam)
        else:
            logging.info(f"Projection already exists for {model_name} with params {hyperparam}")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER,
                    run_id TEXT,
                    vector BLOB,
                    PRIMARY KEY(id, run_id),
                    FOREIGN KEY(run_id) REFERENCES metadata(run_id)
                )
            """)
            cur.execute("SELECT COUNT(*) FROM embeddings WHERE run_id=?", (run_id,))
            count = cur.fetchone()
            exists = count[0] == 5000 if count else False 
            if not exists:
                tasks.append(hyperparam)
            else:
                logging.info(f"Projection is complete")
        conn.close()
    return tasks


def project(args):
    db_path = "/CECI/home/ucl/irec/darimez/embeddings/"
    tables = [
        os.path.join(db_path, f) for f in os.listdir(db_path)
        if f.endswith(".db") and not f=='attention.db'
    ]
    logger = get_logger(args.debug)

    db_path = "/CECI/home/ucl/irec/darimez/proj"
    umap_params = {
        "n_components": [2, 10, 25],
        "n_neighbors": [10, 20, 50, 100],
        "min_dist": [0.0, 0.1, 0.5], 
    }
    tsne_params = {
        "n_components": [2, 10, 25],
        "perplexity": [40],
        "early_exaggeration": [20],
        "learning_rate": [200],
        "max_iter": [1000],
    }

    # Build hyperparam combinations
    def expand_params(param_dict, algo):
        keys = list(param_dict.keys())
        for values in itertools.product(*[param_dict[k] for k in keys]):
            d = dict(zip(keys, values))
            d["algo"] = algo
            yield d

    all_hyperparams = list(expand_params(umap_params, "umap"))
    all_hyperparams += list(
        expand_params(tsne_params, "tsne")
    )
    # Clean up failed/incomplete entries
    # Delete metadata where run_id has no embeddings OR incomplete embeddings (< 5000)
    for db in ["tsne.db", "umap.db"]:
        conn = sqlite3.connect(
            os.path.join(db_path, db)
        )
        c = conn.cursor()
        try:
            c.execute("""
                DELETE FROM metadata 
                WHERE run_id NOT IN (SELECT DISTINCT run_id FROM embeddings)
                OR run_id IN (
                    SELECT run_id FROM embeddings 
                    GROUP BY run_id 
                    HAVING COUNT(*) < 5000
                )
            """)
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
            logger.info(f"Metrics already exist for model {model_name}")
        else:
            logger.info(f"Saving metrics for model {model_name}")
            ids, X, hfs, mious, mconfs, cats, supercats, _ = load_embeddings(table)
            data = (ids, hfs, mious, mconfs, cats, supercats)
            compute_and_store_metrics(model_name, data, db_path=db_path)

        tasks = get_tasks(all_hyperparams, model_name)
        logger.info(
            f"Computing projections for {len(tasks)}/{len(all_hyperparams)} parameters"
        )
        USE_PARALLEL = False
        num_workers = 8
        logger.info(f"USE_PARALLEL: {USE_PARALLEL}, num_workers: {num_workers}, len(tasks): {len(tasks)}")
        if USE_PARALLEL and num_workers > 1 and len(tasks) > 1:
            import time
            
            # Set environment variables to limit threading in child processes
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            
            # Use initializer pattern to pass X to workers without pickle overhead per task
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_worker,
                    initargs=(X, model_name, args)
                ) as executor:
                    # Submit all tasks to executor
                    futures = []
                    for task in tasks:
                        future = executor.submit(compute_projection_task, task)
                        futures.append(future)
                    
                    # Create mapping from future to task
                    future_to_task = {f: t for f, t in zip(futures, tasks)}
                    
                    start_time = time.time()
                    
                    # Process results with tqdm progress bar
                    completed = 0
                    with tqdm(total=len(future_to_task), desc="Projections", unit="task") as pbar:
                        for future in concurrent.futures.as_completed(future_to_task):
                            task = future_to_task[future]
                            completed += 1
                            pbar.update(1)
                            
                            try:
                                result = future.result()
                                if not result.get("success", False):
                                    logger.error(f"Task failed: {task} - Error: {result.get('error', 'Unknown')}")
                                else:
                                    # If worker put embedding on disk, the main process stores it to DB to avoid concurrent writers
                                    tmp_path = result.get('tmp_path')
                                    if tmp_path:
                                        try:
                                            store_projection_from_file(tmp_path, model_name, result.get('run_id'), {k: v for k,v in task.items() if k != 'algo'}, db_path=PROJ_DB_DIR, algo=task['algo'])
                                        except Exception as e:
                                            logger.error(f"Failed to store projection for task {task}: {e}")
                            except concurrent.futures.process.BrokenProcessPool as e:
                                logger.error(f"Process pool was broken for task {task}: {e}")
                            except Exception as e:
                                logger.error(f"Unexpected error for task {task}: {e}")
            except Exception as e:
                logger.error(f"Error in parallel execution, falling back to sequential: {e}")
                # Fallback to sequential execution
                for task in tasks:
                    try:
                        res = compute_projection_task(task)
                        if res and res.get('tmp_path'):
                            try:
                                store_projection_from_file(res.get('tmp_path'), model_name, res.get('run_id'), {k: v for k, v in task.items() if k != 'algo'}, db_path=PROJ_DB_DIR, algo=task['algo'])
                            except Exception as e:
                                logger.error(f"Storing projection failed for task {task}: {e}")
                    except Exception as e:
                        logger.error(f"Sequential execution error for task {task}: {e}")
        elif tasks:
            # Sequential execution
            logger.info("Running projections sequentially") 
            for task in tqdm(tasks, total=len(tasks), desc="Projections", unit="task"):
                if X is None:
                    ids, X, hfs, mious, mconfs, cats, supercats, _ = load_embeddings(table)
                compute_and_store(
                    X,
                    model_name,
                    {k: v for k, v in task.items() if k != "algo"},
                    db_path=db_path,
                    algo=task["algo"],
                )