"""
Quick smoke test for evaluate_clusters.KNN using tiny synthetic embeddings.
Run from repo root:

    python vllmcluster/scripts/smoke_evaluate_clusters.py

This script creates a temporary embeddings DB path (a file), stubs
`load_embeddings` and `get_fold_indices`, sets a temporary results DB,
and calls `KNN(args)` with `args.table` pointing to the fake embeddings DB.
"""
import tempfile
import os
import sqlite3
import numpy as np
import traceback

import importlib.util
import sys
from pathlib import Path

# Ensure the local vllmcluster/utils.py is available as top-level `utils`
repo_root = Path(__file__).resolve().parents[1]
utils_path = repo_root / "utils.py"
spec = importlib.util.spec_from_file_location("utils", str(utils_path))
utils_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_mod)
sys.modules["utils"] = utils_mod

# Import evaluate_clusters from the package file directly
eval_path = repo_root / "evaluate_clusters.py"
spec2 = importlib.util.spec_from_file_location("evaluate_clusters", str(eval_path))
ec = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ec)


class Args:
    def __init__(self, table=None, debug=True):
        self.table = table
        self.debug = debug


def run_smoke():
    tmpdir = tempfile.mkdtemp(prefix="miro-smoke-")
    try:
        # Prepare fake embeddings DB file (presence is enough for KNN table resolution)
        model_db = os.path.join(tmpdir, "model.db")
        open(model_db, "w").close()

        # Prepare temp knn results DB
        knn_db = os.path.join(tmpdir, "knn_results.db")
        ec.DB_PATH = knn_db
        ec.init_db()

        # Small synthetic dataset
        N = 10
        ids = np.arange(N)
        rng = np.random.RandomState(0)
        A = rng.rand(N, N)
        D = (A + A.T) / 2.0
        np.fill_diagonal(D, 0.0)
        hfs = rng.rand(N)
        mious = rng.rand(N)
        mconfs = rng.rand(N)
        cats = rng.randint(0, 2, size=N)
        supercats = rng.randint(0, 2, size=N)

        # Stub load_embeddings used by process_table
        def _load_embeddings_stub(path, query=None):
            if query is not None:
                import pandas as pd
                return pd.DataFrame({query: rng.rand(N)})
            return ids, D, hfs, mious, mconfs, cats, supercats

        ec.load_embeddings = _load_embeddings_stub

        # Stub deterministic folds for N
        def _folds(n_splits=2, random_state=42):
            from sklearn.model_selection import KFold
            idx = np.arange(N)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(kf.split(idx))

        ec.get_fold_indices = _folds

        # Avoid plotting dependency in smoke run
        ec.plot_results = lambda df: "FIG"

        # Call KNN with our single absolute table path
        args = Args(table=model_db, debug=True)
        print("Starting KNN smoke run...")
        ec.KNN(args)

        # Validate DB rows
        conn = sqlite3.connect(ec.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM knn_results")
        cnt = c.fetchone()[0]
        conn.close()

        print(f"Smoke run complete. Rows inserted in knn_results: {cnt}")
        return 0
    except Exception as e:
        traceback.print_exc()
        return 2
    finally:
        # leave temp files for inspection
        pass


if __name__ == "__main__":
    exit(run_smoke())
