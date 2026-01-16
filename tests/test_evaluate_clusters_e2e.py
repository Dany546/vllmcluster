import os
import sqlite3
import pandas as pd
import pytest

import vllmcluster.evaluate_clusters as evalc


class Args:
    def __init__(self, table=None, debug=False):
        self.table = table
        self.debug = debug


@pytest.mark.integration
def test_knn_minimal_end_to_end(tmp_path, tmp_knn_db, dummy_wandb, no_plot, small_embeddings, monkeypatch):
    # Prepare a fake embeddings DB file that KNN() will accept as an absolute path
    model_db = tmp_path / "model.db"
    model_db.write_text("")  # presence is enough for the path checks

    # Monkeypatch get_tasks to return a single short job for our model
    tasks = {
        "model": [
            ("model", 3, 42, 2, "kfold", "euclidean", 2, "hit_freq")
        ]
    }

    monkeypatch.setattr(evalc, "get_tasks", lambda tables, neighbor_grid, RN, num_folds, distance_metric: tasks)

    # Point KNN's DB to our tmp DB and init
    evalc.DB_PATH = tmp_knn_db
    evalc.init_db()

    # Run KNN with our single absolute table path
    args = Args(table=str(model_db), debug=False)
    # Should not raise
    evalc.KNN(args)

    # Validate DB rows
    conn = sqlite3.connect(evalc.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knn_results'")
    assert c.fetchone() is not None, "knn_results table was not created"

    df = pd.read_sql_query("SELECT * FROM knn_results", conn)
    conn.close()

    # At least one row should be inserted
    assert len(df) > 0, "No rows were inserted into knn_results"

    # Metric columns: allow either legacy (with slashes) or sanitized names, assert at least one present
    e2e_metric_cols = {"mae", "mae/accuracy", "accuracy", "r2", "corr", "correlation/ARI", "ARI"}
    present = set(df.columns) & e2e_metric_cols
    assert len(present) >= 1, f"No expected metric columns present, got {df.columns.tolist()}"
