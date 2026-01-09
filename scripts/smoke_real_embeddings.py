"""
Smoke-run using a few rows from real embeddings/proj DBs.
Selects: 1 YOLO, 1 YOLO-seg, 1 DINO, 1 TSNE projection (from proj dir).
Runs only k=5 and a tiny number of folds.

Run from within the project (recommended):

  cd vllmcluster
  PYTHONPATH=.. python scripts/smoke_real_embeddings.py

This script is conservative: it limits loaded embeddings to a few rows to keep memory low.
"""
import os
import sys
import tempfile
import sqlite3
import traceback
from pathlib import Path
import importlib.util

EMB_DIR = "/CECI/home/ucl/irec/darimez/embeddings"
PROJ_DIR = "/CECI/home/ucl/irec/darimez/proj"

# Dynamic import of evaluate_clusters module from this package path
repo_root = Path(__file__).resolve().parents[1]
eval_path = repo_root / "evaluate_clusters.py"
spec = importlib.util.spec_from_file_location("evaluate_clusters", str(eval_path))
ec = importlib.util.module_from_spec(spec)
# Ensure local utils import resolves to vllmcluster/utils.py
utils_path = repo_root / "utils.py"
if utils_path.exists():
    spec_u = importlib.util.spec_from_file_location("utils", str(utils_path))
    utils_mod = importlib.util.module_from_spec(spec_u)
    spec_u.loader.exec_module(utils_mod)
    sys.modules["utils"] = utils_mod

spec.loader.exec_module(ec)


def find_db(pattern, search_dir=EMB_DIR):
    try:
        for fn in os.listdir(search_dir):
            if fn.endswith('.db') and pattern in fn.lower():
                return os.path.join(search_dir, fn)
    except Exception:
        pass
    return None


def find_proj(name, search_dir=PROJ_DIR):
    try:
        for fn in os.listdir(search_dir):
            if fn.endswith('.db') and name in fn.lower():
                return os.path.join(search_dir, fn)
    except Exception:
        pass
    return None


def limit_load_embeddings(orig_loader, max_rows=64):
    def _wrapper(path, query='hit_freq'):
        # Call loader without `query` when not querying a specific column
        if query is None:
            res = orig_loader(path)
        else:
            res = orig_loader(path, query=query)
        # If query is None, expect tuple: ids, X, hfs, mious, mconfs, cats, supercats
        if query is None and isinstance(res, tuple) and len(res) >= 7:
            ids, X, hfs, mious, mconfs, cats, supercats = res[:7]
            # slice
            ids = ids[:max_rows]
            X = X[:max_rows] if hasattr(X, 'shape') else X
            hfs = hfs[:max_rows]
            mious = mious[:max_rows]
            mconfs = mconfs[:max_rows]
            cats = cats[:max_rows]
            supercats = supercats[:max_rows]
            return ids, X, hfs, mious, mconfs, cats, supercats
        # If query requested, return small DataFrame if possible
        return res
    return _wrapper


def run():
    tmpdir = tempfile.mkdtemp(prefix="miro-smoke-")
    knn_db = os.path.join(tmpdir, "knn_results.db")
    ec.DB_PATH = knn_db
    ec.init_db()

    # locate candidate DBs
    yolo = find_db('yolo', EMB_DIR)
    yolo_seg = find_db('yolo', EMB_DIR)
    # prefer seg variant if present
    if yolo_seg and 'seg' not in os.path.basename(yolo_seg).lower():
        yolo_seg = None
    dino = find_db('dino', EMB_DIR)

    tsne_db = find_proj('tsne', PROJ_DIR)

    chosen = {}
    if yolo:
        chosen['yolo'] = yolo
    if yolo_seg:
        chosen['yolo-seg'] = yolo_seg
    else:
        # try to find a filename containing 'seg'
        seg_candidate = find_db('seg', EMB_DIR)
        if seg_candidate:
            chosen['yolo-seg'] = seg_candidate
    if dino:
        chosen['dino'] = dino
    # if tsne_db:
    #     chosen['tsne'] = tsne_db

    if not chosen:
        print("No candidate DBs found under EMB_DIR/PROJ_DIR — aborting.")
        return 2

    print("Selected DBs for smoke run:")
    for k, v in chosen.items():
        print(f" - {k}: {v}")

    # limit rows by wrapping load_embeddings
    try:
        orig_loader = ec.load_embeddings
    except Exception:
        orig_loader = None
    if orig_loader:
        ec.load_embeddings = limit_load_embeddings(orig_loader, max_rows=64)

    # small deterministic folds
    def small_folds(n_splits=2, random_state=42):
        from sklearn.model_selection import KFold
        # assume small N after slicing
        N = 64
        idx = list(range(N))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(idx))

    ec.get_fold_indices = small_folds

    # Force k=5 tasks for chosen tables
    tasks = {}
    for fn in chosen.values():
        table_name = os.path.basename(fn).split('.')[0]
        tasks[table_name] = [
            (table_name, 5, 42, 2, 'kfold', 'euclidean', 2, 'hit_freq')
        ]

    # monkeypatch get_tasks to return our tasks
    ec.get_tasks = lambda tables, neighbor_grid, RN, num_folds, distance_metric: tasks

    # build args.table as comma-separated absolute paths
    table_arg = ",".join(chosen.values())
    class Args:
        def __init__(self, table=None, debug=True):
            self.table = table
            self.debug = debug

    args = Args(table=table_arg, debug=True)

    try:
        print("Starting KNN smoke run (k=5)...")
        ec.KNN(args)
        # check DB
        conn = sqlite3.connect(ec.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM knn_results")
        cnt = c.fetchone()[0]
        conn.close()
        print(f"KNN smoke run finished — rows in knn_results: {cnt}")
        return 0
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    sys.exit(run())
