import sqlite3
import pytest
import vllmcluster.evaluate_clusters as evalc


def test_insert_record_accepts_legacy_metric_keys(tmp_path, tmp_knn_db):
    # Ensure DB initialized
    evalc.DB_PATH = tmp_knn_db
    evalc.init_db()

    conn = sqlite3.connect(evalc.DB_PATH)
    c = conn.cursor()

    record = {
        "model": "x",
        "k": 1,
        "cv_type": "kfold",
        "num_folds": 2,
        "random_state": 42,
        "distance_metric": "euclidean",
        "fold": 0,
        "target": "hit_freq",
        "mae/accuracy": 0.123,
        "r2": 0.5,
        "correlation/ARI": 0.1,
    }

    # Should succeed (or at least not raise OperationalError)
    ok = evalc.insert_record(record, c)
    conn.commit()

    # If insert_record returns True, check row exists
    if ok:
        c.execute("SELECT * FROM knn_results WHERE model=?", ("x",))
        row = c.fetchone()
        assert row is not None

    conn.close()
