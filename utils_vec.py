"""
Compatibility helpers for sql-vector-backed projection databases.

This module provides functions analogous to `utils.py` but which
operate on vec0/vec_projections tables created by `sqlvector_projector.py`.

Keep this separate from `utils.py` so you can maintain a sqlite3-only
legacy branch and a sqlite-vector branch independently.
"""
from typing import Optional
import os
import sqlite3
import numpy as np

from sqlvector_projector import (
    connect_vec_db,
    load_vectors_from_db,
    get_vector_slice,
)


def load_projection_vectors(db_path: str, query: Optional[str] = None) -> np.ndarray:
    """Load projection vectors from a proj DB.

    Args:
        db_path: path to the projection .db file (e.g. /.../proj/umap.db)
        query: optional run identifier in the form `model.algo.hash` or a
               `run_id` string like `model_hash`.

    Returns:
        numpy array shape (n, dim) or empty array when nothing found.
    """
    if not os.path.exists(db_path):
        return np.array([])

    # If caller passed a projection file (e.g., .../umap.db) we need the run id.
    # Accept both `model_hash` and `model.algo.hash` formats.
    run_id = None
    if query is None:
        # try to pick a single run_id from metadata
        conn = connect_vec_db(db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT run_id FROM metadata LIMIT 2")
            rows = cur.fetchall()
            if not rows:
                cur.execute("SELECT DISTINCT run_id FROM vec_projections LIMIT 2")
                rows = cur.fetchall()
            if not rows:
                conn.close()
                return np.array([])
            run_id = rows[0][0]
        finally:
            conn.close()
    else:
        if "." in query:
            parts = query.split(".")
            run_id = parts[0] + "_" + parts[-1]
        else:
            run_id = query

    return load_vectors_from_db(os.path.dirname(db_path) + "/", os.path.basename(db_path).split(".")[0], run_id)


def get_2d_slice(db_path: str, query: str, dims=(0, 1)) -> np.ndarray:
    """Convenience: return only two columns for plotting (UMAP/t-SNE style)."""
    db_dir = os.path.dirname(db_path)
    algo = os.path.basename(db_path).split(".")[0]
    return get_vector_slice(db_dir + "/", algo, query, dims=list(dims))
