"""Database utilities for gridsearch results and fold persistence.

Provides simple helpers to create `grid_search.db`, store folds, and insert
per-fold results. Uses sqlite3 with a small retry loop to handle busy DB.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

DEFAULT_DB_NAME = "grid_search.db"


def get_grid_db_path() -> str:
    # Use $CECIHOME if set, otherwise cwd
    base = os.environ.get('CECIHOME', os.getcwd())
    return os.path.join(base, DEFAULT_DB_NAME)


def _connect(path: str):
    con = sqlite3.connect(path, timeout=30)
    con.execute('PRAGMA journal_mode=WAL;')
    return con


def create_grid_db(path: Optional[str] = None):
    path = path or get_grid_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = _connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS grid_results(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            embedding_model TEXT,
            target TEXT,
            aggregation TEXT,
            preproc TEXT,
            feature_selection TEXT,
            extractor TEXT,
            extractor_params TEXT,
            knn_n INTEGER,
            knn_metric TEXT,
            fold INTEGER,
            spearman REAL,
            r2 REAL,
            mae REAL,
            status TEXT,
            reason TEXT,
            ts REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS folds(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed INTEGER,
            fold_index INTEGER,
            train_idx BLOB,
            val_idx BLOB
        )
        """
    )
    con.commit()
    con.close()


def _insert_with_retry(path: str, sql: str, params: Iterable[Any], max_retries: int = 5, delay: float = 0.1):
    for attempt in range(max_retries):
        try:
            con = _connect(path)
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit()
            con.close()
            return
        except sqlite3.OperationalError:
            time.sleep(delay * (attempt + 1))
    raise


def insert_grid_result(path: Optional[str], row: Dict[str, Any]):
    path = path or get_grid_db_path()
    sql = (
        "INSERT INTO grid_results(run_id, embedding_model, target, aggregation, preproc, feature_selection, extractor, extractor_params, knn_n, knn_metric, fold, spearman, r2, mae, status, reason, ts)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    params = [
        row.get('run_id'),
        row.get('embedding_model'),
        row.get('target'),
        row.get('aggregation'),
        row.get('preproc'),
        row.get('feature_selection'),
        row.get('extractor'),
        row.get('extractor_params'),
        row.get('knn_n'),
        row.get('knn_metric'),
        row.get('fold'),
        row.get('spearman'),
        row.get('r2'),
        row.get('mae'),
        row.get('status', 'ok'),
        row.get('reason'),
        time.time(),
    ]
    _insert_with_retry(path, sql, params)


def has_results(
    path: Optional[str],
    run_id: str,
    embedding_model: Optional[str] = None,
    target: Optional[str] = None,
    preproc: Optional[str] = None,
    feature_selection: Optional[str] = None,
    extractor: Optional[str] = None,
    extractor_params: Optional[str] = None,
) -> bool:
    """Return True if any grid_results rows exist matching the provided keys.

    Only filters on provided non-None parameters; `run_id` is required.
    """
    if run_id is None:
        return False
    path = path or get_grid_db_path()
    clauses = ["run_id = ?"]
    params = [run_id]
    if embedding_model is not None:
        clauses.append("embedding_model = ?")
        params.append(embedding_model)
    if target is not None:
        clauses.append("target = ?")
        params.append(target)
    if preproc is not None:
        clauses.append("preproc = ?")
        params.append(preproc)
    if feature_selection is not None:
        clauses.append("feature_selection = ?")
        params.append(feature_selection)
    if extractor is not None:
        clauses.append("extractor = ?")
        params.append(extractor)
    if extractor_params is not None:
        clauses.append("extractor_params = ?")
        params.append(str(extractor_params))

    q = "SELECT COUNT(1) FROM grid_results WHERE " + " AND ".join(clauses)
    con = _connect(path)
    cur = con.cursor()
    cur.execute(q, params)
    cnt = cur.fetchone()[0]
    con.close()
    return int(cnt) > 0


def insert_skipped(path: Optional[str], info: Dict[str, Any]):
    info = dict(info)
    info['status'] = 'skipped'
    info.setdefault('reason', 'skipped_by_pipeline')
    insert_grid_result(path, info)


def save_folds(path: Optional[str], seed: int, folds: List[Dict[str, Any]]):
    path = path or get_grid_db_path()
    for fi, f in enumerate(folds):
        sql = "INSERT INTO folds(seed, fold_index, train_idx, val_idx) VALUES (?,?,?,?)"
        params = [seed, fi, sqlite3.Binary(np_array_to_bytes(f['train_idx'])), sqlite3.Binary(np_array_to_bytes(f['val_idx']))]
        _insert_with_retry(path, sql, params)


def np_array_to_bytes(arr):
    import numpy as _np

    arr = _np.ascontiguousarray(arr, dtype=_np.int64)
    return arr.tobytes()


def read_embeddings_db(db_path: str, table: str = 'embeddings'):
    """Read embeddings and available targets from a sqlite DB table.

    Returns (X, meta, ids) where X is np.ndarray (n x d), meta is dict of arrays
    for columns found (e.g., 'cls_loss', 'seg_loss'), and ids is list of row ids.
    The table is expected to have columns: id (or rowid), embedding (BLOB of float32),
    and optional numeric columns for targets.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # inspect columns
    cur.execute(f"PRAGMA table_info({table})")
    cols = cur.fetchall()
    if not cols:
        raise RuntimeError(f"Table {table} not found in {db_path}")
    col_names = [c[1] for c in cols]

    # determine embedding column
    emb_col = None
    for c in ['embedding', 'emb', 'features', 'vector']:
        if c in col_names:
            emb_col = c
            break
    if emb_col is None:
        # fallback: assume second column is blob
        emb_col = col_names[1]

    # read all rows
    cur.execute(f"SELECT rowid, * FROM {table}")
    rows = cur.fetchall()
    ids = []
    X_list = []
    meta = {}
    # prepare meta collectors
    for name in col_names:
        if name not in (emb_col,):
            meta[name] = []

    for r in rows:
        # rowid, then columns
        rowid = r[0]
        ids.append(rowid)
        # find embedding value
        # locate index of emb_col in col_names
        emb_idx = col_names.index(emb_col)
        emb_blob = r[1 + emb_idx]
        if emb_blob is None:
            X_list.append(None)
        else:
            arr = np.frombuffer(emb_blob, dtype=np.float32)
            X_list.append(arr)
        # collect other columns
        for i, name in enumerate(col_names):
            if name == emb_col:
                continue
            val = r[1 + i]
            meta[name].append(val)

    # stack embeddings, handle variable lengths
    lens = [x.shape[0] if x is not None else 0 for x in X_list]
    if len(set(lens)) != 1:
        raise RuntimeError("Embeddings have varying lengths; cannot stack")
    X = np.vstack(X_list).astype(np.float32)
    # convert meta to arrays
    for k in list(meta.keys()):
        try:
            meta[k] = np.array(meta[k])
        except Exception:
            meta[k] = np.array(meta[k], dtype=object)

    con.close()
    return X, meta, ids


class DistancesDBReader:
    """Reader for distances stored in a sqlite DB with table `distances(i,j,dist)`.

    Methods reconstruct pairwise matrices for given id lists.
    """

    def __init__(self, db_path: str, table: str = 'distances'):
        self.db_path = db_path
        self.table = table

    def get_pairwise_distance_matrix(self, ids: List[int]) -> np.ndarray:
        """Return an n x n matrix of distances for the given ids order."""
        n = len(ids)
        id_to_idx = {int(i): idx for idx, i in enumerate(ids)}
        D = np.zeros((n, n), dtype=np.float32)
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        # read rows where i or j in ids
        q = f"SELECT i,j,dist FROM {self.table} WHERE i IN ({','.join('?'*n)}) AND j IN ({','.join('?'*n)})"
        params = [int(x) for x in ids] + [int(x) for x in ids]
        cur.execute(q, params)
        for i, j, d in cur.fetchall():
            if int(i) in id_to_idx and int(j) in id_to_idx:
                D[id_to_idx[int(i)], id_to_idx[int(j)]] = float(d)
        con.close()
        return D

    def get_cross_distance_matrix(self, test_ids: List[int], train_ids: List[int]) -> np.ndarray:
        """Return an n_test x n_train matrix of distances from test to train."""
        n_t = len(test_ids)
        n_tr = len(train_ids)
        train_map = {int(i): idx for idx, i in enumerate(train_ids)}
        test_map = {int(i): idx for idx, i in enumerate(test_ids)}
        D = np.zeros((n_t, n_tr), dtype=np.float32)
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        q = f"SELECT i,j,dist FROM {self.table} WHERE i IN ({','.join('?'*n_t)}) AND j IN ({','.join('?'*n_tr)})"
        params = [int(x) for x in test_ids] + [int(x) for x in train_ids]
        cur.execute(q, params)
        for i, j, d in cur.fetchall():
            if int(i) in test_map and int(j) in train_map:
                D[test_map[int(i)], train_map[int(j)]] = float(d)
        con.close()
        return D
