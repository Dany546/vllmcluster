# project_refactor.py
import os
import itertools
import sqlite3
from sqlvector_projector import (
    connect_vec_db,
    create_vec_tables,
    compute_and_store,
    compute_and_store_metrics,
    load_vectors_from_db,
    get_vector_slice,
    compute_run_id,
)
from utils import get_logger, load_embeddings, get_lookups

# Configure paths
EMBEDDINGS_DIR = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
PROJ_DB_DIR = "/globalscratch/ucl/irec/darimez/dino/proj/"
METRICS_DB = os.path.join(PROJ_DB_DIR, "metrics.db")

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
    conn = sqlite3.connect(db_file)
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
    conn = sqlite3.connect(db_file)
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
        "n_neighbors": [15, 30, 60],
        "min_dist": [0.02, 0.1, 0.5],
        "n_components": [2, 10, 50],
    }
    tsne_params = {
        "n_components": [2, 10, 50],
        "perplexity": [10, 30, 50],
        "early_exaggeration": [8, 12, 16],
        "learning_rate": [100, 200, 500],
        "max_iter": [1000],
    }

    all_hyperparams = list(expand_params(umap_params, "umap")) + list(expand_params(tsne_params, "tsne"))

    # Clean up orphan metadata entries in projection DBs (optional)
    for dbname in ("tsne.db", "umap.db"):
        dbfile = os.path.join(PROJ_DB_DIR, dbname)
        if not os.path.exists(dbfile):
            continue
        conn = sqlite3.connect(dbfile)
        c = conn.cursor()
        try:
            c.execute("""DELETE FROM metadata WHERE run_id NOT IN (SELECT DISTINCT run_id FROM vec_projections);""")
        except Exception:
            # ignore if vec_projections doesn't exist
            pass
        conn.commit()
        conn.close()

    # iterate over embedding files (one file per model)
    tables = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".db")]
    for table in tables:
        model_name = os.path.basename(table).split(".")[0]
        logger.debug(f"Processing {model_name}")

        # 1) compute and store metrics if missing
        # check metrics DB for this model
        conn = sqlite3.connect(METRICS_DB)
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
        for task in tasks:
            algo = task["algo"]
            params = {k: v for k, v in task.items() if k != "algo"}
            logger.info(f"Computing {algo} projection for model {model_name} with params {params}")
            # compute_and_store will create vec tables and metadata and insert projections
            compute_and_store(X, model_name, params, db_path=PROJ_DB_DIR, algo=algo)

    logger.info("Projection pipeline finished.")
