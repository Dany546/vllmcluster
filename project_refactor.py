# project_refactor.py
import os
import itertools
import json
import concurrent.futures
from tqdm import tqdm
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
from utils import get_logger, load_embeddings, get_lookups

# Configure paths
EMBEDDINGS_DIR = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
PROJ_DB_DIR = "/CECI/home/ucl/irec/darimez" # "/globalscratch/ucl/irec/darimez/dino/proj/"
METRICS_DB = os.path.join(PROJ_DB_DIR, "metrics.db")
# Parallel execution configuration
USE_PARALLEL = True  # Set to False to disable parallel execution for debugging
MAX_WORKERS = 8   # Reduced from 24 to prevent OOM - UMAP is memory intensive

# Memory management
MAX_MEMORY_PERCENT = 70  # Max percent of available memory to use


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


def expand_params(param_dict, algo):
    keys = list(param_dict.keys())
    for values in itertools.product(*[param_dict[k] for k in keys]):
        d = dict(zip(keys, values))
        d["algo"] = algo
        yield d

def run_exists_in_db(run_id: str, algo: str) -> bool:
    db_file = os.path.join(PROJ_DB_DIR, f"{algo}.db")
    if not os.path.exists(db_file):
        return False
    conn = connect_vec_db(db_file)
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM metadata WHERE run_id=?", (run_id,))
        exists = cur.fetchone() is not None
    except Exception:
        exists = False
    conn.close()
    return exists

def embeddings_count_for_run(run_id: str, algo: str) -> int:
    db_file = os.path.join(PROJ_DB_DIR, f"{algo}.db")
    if not os.path.exists(db_file):
        return 0
    conn = connect_vec_db(db_file)
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM vec_projections WHERE run_id=?", (run_id,))
        count = cur.fetchone()[0]
    except Exception:
        # fallback to embeddings table if vec_projections not present
        try:
            cur.execute("SELECT COUNT(*) FROM embeddings WHERE run_id=?", (run_id,))
            count = cur.fetchone()[0]
        except Exception:
            count = 0
    conn.close()
    return count

def project(args):
    logger = get_logger(args.debug)

    # hyperparameter grids (same as original) 
    umap_params = {
        "n_components": [2, 10, 25],
        "n_neighbors": [10, 20, 50, 100],
        "min_dist": [0.0, 0.1, 0.5], 
    }
    tsne_params = {
        "n_components": [2],
        "perplexity": [40],
        "early_exaggeration": [20],
        "learning_rate": [200],
        "max_iter": [1000],
    }

    all_hyperparams = list(expand_params(umap_params, "umap")) 
    all_hyperparams += list(expand_params(tsne_params, "tsne"))

    # Preflight check: ensure vec0 is available (fail fast if not)
    try:
        # Try connecting requiring vec0 — this will raise if vec0 isn't available
        _ = connect_vec_db(os.path.join(PROJ_DB_DIR, "tsne.db"), require_vec0=True)
        _.close()
    except Exception as e:
        msg = (
            "vec0 (sqlite-vec) is not available in the current Python environment.\n"
            "Install it into your environment (e.g., `pip install sqlite-vec`) or set `VECTOR_EXT_PATH`/`VEC0_SO` to the vec0 .so path.\n"
            f"Error: {e}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Clean up orphan metadata entries in projection DBs (optional)
    for dbname in ("tsne.db", "umap.db"):
        dbfile = os.path.join(PROJ_DB_DIR, dbname)
        if not os.path.exists(dbfile):
            continue
        conn = connect_vec_db(dbfile)
        c = conn.cursor()
        try:
            c.execute("""DELETE FROM metadata WHERE run_id NOT IN (SELECT DISTINCT run_id FROM vec_projections);""")
        except Exception:
            # ignore if vec_projections doesn't exist
            pass
        conn.close()

    # iterate over embedding files (one file per model)
    tables = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".db")]
    print(len(tables), "embedding tables found.")
    for table in tables:  # process only the last table for now
        model_name = os.path.basename(table).split(".")[0]
        logger.debug(f"Processing {model_name}")

        # 1) compute and store metrics if missing
        # check metrics DB for this model
        conn = connect_vec_db(METRICS_DB)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM vec_metrics WHERE model=?", (model_name,))
            exists_metrics = cur.fetchone()[0] >= 5000
        except Exception:
            exists_metrics = False
        conn.close()

        if exists_metrics:
            logger.info(f"Metrics already exist for model {model_name}, skipping metrics computation")
        else:
            logger.info(f"Computing and storing metrics for model {model_name}")
            ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
            compute_and_store_metrics(model_name, (ids, hfs, mious, mconfs, cats, supercats), db_path=PROJ_DB_DIR)

        # 2) determine which projection tasks are needed for this model
        tasks = []
        for hyperparam in all_hyperparams:
            algo = hyperparam["algo"]
            run_id = compute_run_id(model_name, {k: v for k, v in hyperparam.items() if k != "algo"})
            # if run metadata exists and embeddings count is sufficient, skip
            count = embeddings_count_for_run(run_id, algo)
            if count >= 5000:
                logger.debug(f"Run {run_id} for {algo} already has {count} embeddings; skipping")
                continue
            tasks.append(hyperparam)

        logger.info(f"Computing projections for {len(tasks)}/{len(all_hyperparams)} parameter combinations for model {model_name}")

        # 3) compute projections for each missing task
        # load embeddings once per model and reuse
        ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
        
        # Determine number of workers based on configuration
        num_workers = MAX_WORKERS
        
        # Ensure tasks are processed in deterministic order by sorting them 
        tasks_sorted = sorted(tasks, key=lambda x: (x["algo"], json.dumps(x, sort_keys=True)))
        
        logger.info(f"USE_PARALLEL: {USE_PARALLEL}, num_workers: {num_workers}, len(tasks_sorted): {len(tasks_sorted)}")
        if USE_PARALLEL and num_workers > 1 and len(tasks_sorted) > 1:
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
                    for task in tasks_sorted:
                        future = executor.submit(compute_projection_task, task)
                        futures.append(future)
                    
                    # Create mapping from future to task
                    future_to_task = {f: t for f, t in zip(futures, tasks_sorted)}
                    
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
                for task in tasks_sorted:
                    try:
                        res = compute_projection_task(task)
                        if res and res.get('tmp_path'):
                            try:
                                store_projection_from_file(res.get('tmp_path'), model_name, res.get('run_id'), {k: v for k, v in task.items() if k != 'algo'}, db_path=PROJ_DB_DIR, algo=task['algo'])
                            except Exception as e:
                                logger.error(f"Storing projection failed for task {task}: {e}")
                    except Exception as e:
                        logger.error(f"Sequential execution error for task {task}: {e}")
        elif tasks_sorted:
            # Sequential execution
            logger.info("Running projections sequentially")
            for task in tqdm(tasks_sorted, total=len(tasks_sorted), desc="Projections", unit="task"):
                try:
                    _init_worker(X, model_name, args)
                    res = compute_projection_task(task)
                    if res and res.get('tmp_path'):
                        try:
                            store_projection_from_file(res.get('tmp_path'), model_name, res.get('run_id'), {k: v for k, v in task.items() if k != 'algo'}, db_path=PROJ_DB_DIR, algo=task['algo'])
                        except Exception as e:
                            logger.error(f"Storing projection failed for task {task}: {e}")
                except Exception as e:
                    logger.error(f"Sequential execution error for task {task}: {e}")

    logger.info("Projection pipeline finished.")
