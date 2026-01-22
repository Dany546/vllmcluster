import sqlite3
import numpy as np
import pandas as pd
import pytest

import vllmcluster.evaluate_clusters as evalc


def make_sym_distance(N, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.rand(N, N)
    D = (A + A.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def test_work_fn_regression_small(tmp_knn_db):
    evalc.DB_PATH = tmp_knn_db
    evalc.init_db()

    N = 8
    D = make_sym_distance(N, seed=1)
    y = np.linspace(0, 1, N)
    folds = [(np.array([0,1,2,3]), np.array([4,5,6,7])), (np.array([4,5,6,7]), np.array([0,1,2,3]))]

    args = (
        D,
        y,
        "reg_model",
        3,
        42,
        2,
        "kfold",
        "euclidean",
        2,
        "hit_freq",
        folds,
    )

    records = evalc.work_fn(args)
    assert len(records) == len(folds)

    conn = sqlite3.connect(evalc.DB_PATH)
    df = pd.read_sql_query("SELECT * FROM knn_results WHERE model='reg_model'", conn)
    conn.close()
    assert len(df) == len(folds)
    # Check regression metrics exist (r2 present and finite)
    assert "r2" in df.columns
    assert df["r2"].notnull().all()


def test_work_fn_classification_small(tmp_knn_db):
    evalc.DB_PATH = tmp_knn_db
    evalc.init_db()

    N = 8
    D = make_sym_distance(N, seed=2)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = [(np.array([0,1,2,3]), np.array([4,5,6,7])), (np.array([4,5,6,7]), np.array([0,1,2,3]))]

    args = (
        D,
        y,
        "clf_model",
        1,
        42,
        2,
        "kfold",
        "euclidean",
        2,
        "flag_cat",
        folds,
    )

    records = evalc.work_fn(args)
    assert len(records) == len(folds)

    conn = sqlite3.connect(evalc.DB_PATH)
    df = pd.read_sql_query("SELECT * FROM knn_results WHERE model='clf_model'", conn)
    conn.close()
    assert len(df) == len(folds)
    # Check classification metric presence (accuracy or mae/accuracy)
    assert any(col in df.columns for col in ["mae/accuracy", "mae", "accuracy"])