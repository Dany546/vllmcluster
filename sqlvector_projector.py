# sqlvector_projector.py
import os
import json
import hashlib
import itertools
import sqlite3
from typing import List, Tuple, Any, Optional

import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

# Optional helper package that registers vec0 and helpers
try:
    import sqlite_vec
except Exception:
    sqlite_vec = None

# ---------- Helpers ----------
def connect_vec_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-20000")
    # Try to register sqlite-vec via its Python loader if available
    if sqlite_vec is not None:
        try:
            sqlite_vec.load(conn)
            # verify extension loaded
            conn.execute("SELECT vec_version()").fetchone()
        except Exception:
            # ignore and fall back to linear scan behavior
            pass
    return conn

def serialize_float32_array(arr: np.ndarray) -> bytes:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if sqlite_vec is not None and hasattr(sqlite_vec, "serialize_float32"):
        try:
            return sqlite_vec.serialize_float32(arr.tolist())
        except Exception:
            pass
    return arr.tobytes()

def deserialize_float32_array(blob: bytes, dim: int) -> np.ndarray:
    if sqlite_vec is not None and hasattr(sqlite_vec, "deserialize_float32"):
        try:
            arr = np.array(sqlite_vec.deserialize_float32(blob), dtype=np.float32)
            if arr.size != dim:
                raise ValueError(f"Expected {dim} dims, got {arr.size}")
            return arr
        except Exception:
            pass
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} dims, got {arr.size}")
    return arr

def compute_run_id(model: str, hyperparams: dict) -> str:
    hp_items = sorted(hyperparams.items())
    hp_str = json.dumps(hp_items)
    base = f"{model}:{hp_str}"
    run_hash = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{model}_{run_hash}"

# ---------- Schema creation ----------
def create_vec_tables(conn: sqlite3.Connection, proj_dim: int, metrics_dim: int):
    """
    Create vec0 virtual tables for projections and metrics.
    Keep a small metadata table for run-level human-readable info.
    """
    cur = conn.cursor()

    # metadata table (human readable)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            run_id TEXT PRIMARY KEY,
            model TEXT,
            params TEXT
        )
    """)

    # vec_projections: store projection vectors and frequently filtered columns
    # embedding dimension = proj_dim
    try:
        cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_projections USING vec0(id UNINDEXED, run_id UNINDEXED, model UNINDEXED, mean_conf, embedding FLOAT[{proj_dim}])")
    except sqlite3.OperationalError:
        # vec0 not available; create a fallback table with BLOB
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vec_projections (
                id INTEGER,
                run_id TEXT,
                model TEXT,
                mean_conf REAL,
                embedding BLOB,
                PRIMARY KEY(id, run_id)
            )
        """)

    # vec_metrics: store per-image metric vectors (hit_freq, mean_iou, mean_conf, cat counts..., supercat counts...)
    try:
        cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_metrics USING vec0(img_id UNINDEXED, model UNINDEXED, embedding FLOAT[{metrics_dim}])")
    except sqlite3.OperationalError:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vec_metrics (
                img_id INTEGER,
                model TEXT,
                embedding BLOB,
                PRIMARY KEY(model, img_id)
            )
        """)

    conn.commit()

# ---------- Insert / store projections ----------
def insert_projection_batch(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]], proj_dim: int):
    """
    rows: list of tuples (id, run_id, model, mean_conf, embedding_numpy_array)
    If vec0 is available we insert serialized float32 into vec_projections.embedding.
    """
    cur = conn.cursor()

    # Detect whether vec_projections is virtual or fallback by checking schema
    cur.execute("PRAGMA table_info(vec_projections)")
    cols = [r[1] for r in cur.fetchall()]
    is_virtual = 'embedding' in cols and any(c.lower() == 'embedding' for c in cols) and sqlite_vec is not None

    if is_virtual:
        # Insert serialized bytes directly; vec0 will accept the BLOB format
        prepared = []
        for id_, run_id, model, mean_conf, emb in rows:
            prepared.append((int(id_), run_id, model, float(mean_conf), serialize_float32_array(emb)))
        cur.executemany("INSERT OR REPLACE INTO vec_projections(id, run_id, model, mean_conf, embedding) VALUES (?, ?, ?, ?, ?)", prepared)
    else:
        # fallback table with BLOB column
        prepared = []
        for id_, run_id, model, mean_conf, emb in rows:
            prepared.append((int(id_), run_id, model, float(mean_conf), serialize_float32_array(emb)))
        cur.executemany("INSERT OR REPLACE INTO vec_projections(id, run_id, model, mean_conf, embedding) VALUES (?, ?, ?, ?, ?)", prepared)

    conn.commit()

def compute_and_store(X, model_name, hyperparams, db_path="", algo="tsne"):
    assert algo in ["tsne", "umap"], "Unknown algorithm"

    run_id = compute_run_id(model_name, hyperparams)

    if algo == "tsne":
        projector = TSNE(**hyperparams, random_state=42, method="exact")
    else:
        projector = UMAP(**hyperparams, random_state=42)

    embedding = projector.fit_transform(X).astype(np.float32)

    # open DB and ensure vec tables exist
    db_file = os.path.join(db_path, f"{algo}.db")
    conn = connect_vec_db(db_file)

    # create vec tables with projection dimension
    # metrics_dim is unknown here; pass a placeholder (0) — metrics table should be created elsewhere with correct dim
    create_vec_tables(conn, proj_dim=embedding.shape[1], metrics_dim=0)

    # store run metadata
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO metadata(run_id, model, params) VALUES (?, ?, ?)", (run_id, model_name, json.dumps(hyperparams)))
    conn.commit()

    # insert embeddings in batches
    batch = []
    for i, vec in enumerate(embedding):
        # mean_conf not available here; store 0.0 as placeholder or compute if you have it
        batch.append((int(i), run_id, model_name, 0.0, vec))
        if len(batch) >= 128:
            insert_projection_batch(conn, batch, proj_dim=embedding.shape[1])
            batch = []
    if batch:
        insert_projection_batch(conn, batch, proj_dim=embedding.shape[1])

    conn.close()
    return run_id

# ---------- Load vectors ----------
def load_vectors_from_db(db_path, algo, run_id, ids=None) -> np.ndarray:
    db_file = os.path.join(db_path, f"{algo}.db")
    conn = connect_vec_db(db_file)
    cur = conn.cursor()

    # Try to read from vec_projections; if virtual table exists we can select embedding directly
    try:
        if ids is None:
            cur.execute("SELECT id, embedding FROM vec_projections WHERE run_id=? ORDER BY id", (run_id,))
        else:
            placeholders = ",".join("?" * len(ids))
            cur.execute(f"SELECT id, embedding FROM vec_projections WHERE run_id=? AND id IN ({placeholders}) ORDER BY id", (run_id, *ids))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # fallback: maybe vec_projections is a fallback table or vec0 not available
        if ids is None:
            cur.execute("SELECT id, embedding FROM vec_projections WHERE run_id=? ORDER BY id", (run_id,))
        else:
            placeholders = ",".join("?" * len(ids))
            cur.execute(f"SELECT id, embedding FROM vec_projections WHERE run_id=? AND id IN ({placeholders}) ORDER BY id", (run_id, *ids))
        rows = cur.fetchall()

    conn.close()
    if not rows:
        return np.array([])

    # infer dim from first blob
    first_blob = rows[0][1]
    dim = len(np.frombuffer(first_blob, dtype=np.float32))
    vectors = []
    for _id, blob in rows:
        vec = deserialize_float32_array(blob, dim)
        vectors.append(vec)
    return np.vstack(vectors)

def get_vector_slice(db_path, algo, run_id, ids=None, dims=[0, 1]):
    vectors = load_vectors_from_db(db_path, algo, run_id, ids)
    if vectors.size == 0:
        return np.array([])
    return vectors[:, dims]

# ---------- Metrics storage as vector ----------
def compute_and_store_metrics(model_name, data, db_path=""):
    """
    Build a numeric vector per image containing:
      [hit_freq, mean_iou, mean_conf, cat_counts..., supercat_counts...]
    and store it in vec_metrics for fast filtering and similarity.
    """
    from utils import get_lookups  # keep your existing helper
    anns_by_image, id_to_name, id_to_super, categories, supercategories = get_lookups()
    ids, hfs, mious, mconfs, cats, supercats = data

    # compute metrics vector dimension
    metrics_dim = 3 + len(categories) + len(supercategories)  # hit_freq, mean_iou, mean_conf + counts

    db_file = os.path.join(db_path, "metrics.db")
    conn = connect_vec_db(db_file)
    create_vec_tables(conn, proj_dim=0, metrics_dim=metrics_dim)  # ensure vec_metrics exists

    cur = conn.cursor()

    rows = []
    for img_id, hf, miou, mconf, cat, supcat in zip(ids, hfs, mious, mconfs, cats, supercats):
        anns = anns_by_image[img_id]
        cat_counts = {name: 0 for name in categories}
        super_counts = {sc: 0 for sc in supercategories}
        for ann in anns:
            cat_name = id_to_name[ann["category_id"]]
            supercat = id_to_super[ann["category_id"]]
            cat_counts[cat_name] += 1
            super_counts[supercat] += 1

        vec = np.zeros(metrics_dim, dtype=np.float32)
        vec[0] = float(hf)
        vec[1] = float(miou)
        vec[2] = float(mconf)
        vec[3:3+len(categories)] = np.array(list(cat_counts.values()), dtype=np.float32)
        vec[3+len(categories):] = np.array(list(super_counts.values()), dtype=np.float32)

        rows.append((int(img_id), model_name, serialize_float32_array(vec)))

        if len(rows) >= 128:
            cur.executemany("INSERT OR REPLACE INTO vec_metrics(img_id, model, embedding) VALUES (?, ?, ?)", rows)
            rows = []
            conn.commit()

    if rows:
        cur.executemany("INSERT OR REPLACE INTO vec_metrics(img_id, model, embedding) VALUES (?, ?, ?)", rows)
        conn.commit()

    conn.close()

# ---------- kNN query example using vec0 ----------
def knn_query_projections(db_path, algo, query_vec: np.ndarray, k: int = 10, run_id: Optional[str] = None, min_conf: Optional[float] = None):
    db_file = os.path.join(db_path, f"{algo}.db")
    conn = connect_vec_db(db_file)
    cur = conn.cursor()
    qblob = serialize_float32_array(query_vec.astype(np.float32))

    where_clauses = []
    params = [qblob]
    if run_id is not None:
        where_clauses.append("run_id = ?")
        params.append(run_id)
    if min_conf is not None:
        where_clauses.append("mean_conf >= ?")
        params.append(float(min_conf))

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"""
        SELECT id, run_id, model, mean_conf, vec_distance_L2(embedding, ?) AS dist
        FROM vec_projections
        {where_sql}
        ORDER BY dist
        LIMIT ?
    """
    params.append(k)
    try:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # vec0 not available: fallback to linear scan
        # read all matching rows and compute distances in Python
        if run_id is None:
            cur.execute("SELECT id, run_id, model, mean_conf, embedding FROM vec_projections")
        else:
            cur.execute("SELECT id, run_id, model, mean_conf, embedding FROM vec_projections WHERE run_id=?", (run_id,))
        all_rows = cur.fetchall()
        if not all_rows:
            conn.close()
            return []
        # infer dim
        dim = len(np.frombuffer(all_rows[0][4], dtype=np.float32))
        q = query_vec.astype(np.float32)
        scored = []
        for r in all_rows:
            emb = deserialize_float32_array(r[4], dim)
            dist = float(np.linalg.norm(q - emb))
            scored.append((dist, r))
        scored.sort(key=lambda x: x[0])
        rows = [(*r[1][:4], d) for d, r in scored[:k]]  # format similar to vec0 result
    conn.close()
    return rows
