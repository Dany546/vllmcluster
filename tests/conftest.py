import os
import sqlite3
import types
import numpy as np
import pytest
import importlib.util
import sys
from pathlib import Path

# Ensure local vllmcluster/utils.py satisfies top-level `utils` imports in the module under test
utils_path = Path(__file__).resolve().parents[1] / "utils.py"
spec = importlib.util.spec_from_file_location("utils", str(utils_path))
utils_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_mod)
sys.modules["utils"] = utils_mod

import vllmcluster.evaluate_clusters as evalc


class DummyRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        pass


@pytest.fixture
def tmp_knn_db(tmp_path, monkeypatch):
    db_path = tmp_path / "knn_results.db"
    # Ensure file exists
    db_path.write_text("")
    monkeypatch.setattr(evalc, "DB_PATH", str(db_path))
    return str(db_path)


@pytest.fixture
def dummy_wandb(monkeypatch):
    run = DummyRun()

    def fake_init(**kwargs):
        return run

    monkeypatch.setattr(evalc.wandb, "init", fake_init)

    def fake_table(data=None, columns=None):
        return {"data": data, "columns": columns}

    monkeypatch.setattr(evalc.wandb, "Table", fake_table)
    return run


@pytest.fixture
def no_plot(monkeypatch):
    # Replace plot_results with a simple stub to avoid plotly dependencies
    monkeypatch.setattr(evalc, "plot_results", lambda df: "FIGURE")
    return lambda df: "FIGURE"


@pytest.fixture
def small_embeddings(monkeypatch):
    """Provide a small synthetic embedding/distances and metadata for tests.
    Returns a tuple (N, ids, distances, hfs, mious, mconfs, cats, supercats)
    """
    N = 12
    ids = np.arange(N)
    # Small symmetric distance matrix with zeros on diagonal
    rng = np.random.RandomState(0)
    A = rng.rand(N, N)
    D = (A + A.T) / 2
    np.fill_diagonal(D, 0.0)

    hfs = rng.rand(N)
    mious = rng.rand(N)
    mconfs = rng.rand(N)
    cats = rng.randint(0, 2, size=N)
    supercats = rng.randint(0, 2, size=N)

    def _loader(path, query=None):
        # process_table expects load_embeddings(table_path) -> ids, X, hfs, mious, mconfs, cats, supercats
        if query is None:
            return ids, D, hfs, mious, mconfs, cats, supercats
        else:
            # return a tiny DataFrame-like object with the requested column
            import pandas as pd

            return pd.DataFrame({query: rng.rand(N)})

    monkeypatch.setattr(evalc, "load_embeddings", _loader)
    # Also replace get_fold_indices to work on length N rather than hardcoded 5000
    def folds_for_N(n_splits=2, random_state=42):
        from sklearn.model_selection import KFold

        idx = np.arange(N)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(kf.split(idx))

    monkeypatch.setattr(evalc, "get_fold_indices", folds_for_N)

    return N
