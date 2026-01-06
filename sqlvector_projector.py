# sqlvector_projector.py
import os
import json
import hashlib
import itertools
import sys
from typing import List, Tuple, Any, Optional

import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

# Use APSW as the primary method
try:
    import apsw
except Exception:
    apsw = None

# Type alias for connection to work with both APSW and sqlite3
Connection = Any

# ---------- Helpers ----------
def connect_vec_db(db_path: str, require_vec0: bool = False) -> Connection:
    """Connect to SQLite using APSW and load sqlite-vec extension if available.

    If `require_vec0` is True this function raises if vec0 can't be loaded.
    """
    if apsw is None:
        raise RuntimeError('APSW is required. Install via `pip install apsw`')
    
    # Use APSW as the primary method for extension loading
    try:
        aconn = apsw.Connection(db_path)
        
        # Set PRAGMA settings for performance and concurrent access
        try:
            aconn.execute("PRAGMA journal_mode=WAL")
            aconn.execute("PRAGMA synchronous=NORMAL")
            aconn.execute("PRAGMA temp_store=MEMORY")
            aconn.execute("PRAGMA cache_size=-20000")
            # Set busy timeout to handle contention from multiple processes
            # This prevents "database is locked" errors under concurrent access
            aconn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
        except Exception:
            pass
        
        # Try to load vec0 extension if available
        vecpath = os.environ.get("VECTOR_EXT_PATH") or os.environ.get("VEC0_SO")
        if vecpath and os.path.exists(vecpath):
            try:
                aconn.loadextension(vecpath)
                # Verify vec0 is loaded
                try:
                    aconn.execute("SELECT vec_version()")
                    return aconn
                except Exception:
                    # vec0 extension loaded but vec_version() not available
                    return aconn
            except Exception as e:
                if require_vec0:
                    raise RuntimeError(f"Failed to load vec0 extension via APSW: {e}")
                # Continue without vec0 if not required
                return aconn
        elif require_vec0:
            raise RuntimeError('vec0 shared library not found; set VECTOR_EXT_PATH or VEC0_SO to the vec0 .so path')
        
        return aconn
    except Exception as e:
        raise RuntimeError(f"Failed to connect via APSW: {e}")

def serialize_float32_array(arr: np.ndarray) -> bytes:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # Use numpy's tobytes() method for serialization
    return arr.tobytes()

def deserialize_float32_array(blob: bytes, dim: int) -> np.ndarray:
    # Accept bytes or array-like objects (sqlite-vec may return arrays)
    if isinstance(blob, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(blob, dtype=np.float32)
    else:
        arr = np.array(blob, dtype=np.float32)
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
def create_vec_tables(conn: Connection, proj_dim: int, metrics_dim: int):
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
    except Exception:
        # vec0 not available or vec0 constructor incompatible; create a fallback table with BLOB
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
    except Exception:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vec_metrics (
                img_id INTEGER,
                model TEXT,
                embedding BLOB,
                PRIMARY KEY(model, img_id)
            )
        """)

    # Commit for both sqlite3 and APSW connections
    try:
        conn.commit()
    except Exception:
        try:
            conn.execute('COMMIT')
        except Exception:
            pass

# ---------- Insert / store projections ----------
def insert_projection_batch(conn: Connection, rows: List[Tuple[Any, ...]], proj_dim: int):
    """
    rows: list of tuples (id, run_id, model, mean_conf, embedding_numpy_array)
    If vec0 is available we insert serialized float32 into vec_projections.embedding.
    """
    cur = conn.cursor()

    # Detect whether vec_projections is virtual or fallback by checking schema
    cur.execute("PRAGMA table_info(vec_projections)")
    cols = [r[1] for r in cur.fetchall()]
    is_virtual = 'embedding' in cols and any(c.lower() == 'embedding' for c in cols)

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

    # Commit for both sqlite3 and APSW connections
    try:
        conn.commit()
    except Exception:
        try:
            conn.execute('COMMIT')
        except Exception:
            pass

def compute_and_store(X, model_name, hyperparams, db_path="", algo="tsne"):
    assert algo in ["tsne", "umap"], "Unknown algorithm"
    
    # Set environment variables to prevent threading issues in subprocesses
    import os
    import time
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    
    run_id = compute_run_id(model_name, hyperparams)

    if algo == "tsne":
        projector = TSNE(**hyperparams, random_state=42, method="exact")
    else:
        # For UMAP, set n_jobs=1 and disable threading to prevent conflicts with process pool
        umap_params = hyperparams.copy()
        umap_params.setdefault('n_jobs', 1)
        # Set umap._init_graph = False to avoid internal threading
        projector = UMAP(**umap_params, random_state=42)

    embedding = projector.fit_transform(X).astype(np.float32)

    # open DB and ensure vec tables exist
    db_file = os.path.join(db_path, f"{algo}.db")
    
    # Retry logic for database operations (handles concurrent access corruption)
    max_retries = 3
    for retry in range(max_retries):
        try:
            conn = connect_vec_db(db_file)
            
            # create vec tables with projection dimension
            # metrics_dim is unknown here; pass a placeholder (0) — metrics table should be created elsewhere with correct dim
            create_vec_tables(conn, proj_dim=embedding.shape[1], metrics_dim=0)

            # store run metadata
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO metadata(run_id, model, params) VALUES (?, ?, ?)", (run_id, model_name, json.dumps(hyperparams)))
            try:
                conn.commit()
            except Exception:
                try:
                    conn.execute('COMMIT')
                except Exception:
                    pass

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
            
        except Exception as e:
            error_str = str(e).lower()
            if "not a database" in error_str or "corrupted" in error_str or "malformed" in error_str:
                # Database is corrupted, try to recover
                if retry < max_retries - 1:
                    try:
                        if 'conn' in locals():
                            conn.close()
                    except Exception:
                        pass
                    
                    # Try to recover using integrity check and salvage
                    if os.path.exists(db_file):
                        try:
                            # Try to dump good records using sqlite3
                            import sqlite3
                            temp_db = f"{db_file}.recovery.db"
                            
                            # First, try integrity check
                            temp_conn = sqlite3.connect(db_file)
                            temp_cur = temp_conn.cursor()
                            temp_cur.execute("PRAGMA integrity_check")
                            integrity = temp_cur.fetchone()[0]
                            
                            if integrity == "ok":
                                temp_conn.close()
                            else:
                                # Try to dump to new database
                                temp_conn.execute(f"BEGIN;")
                                temp_conn.execute(f"PRAGMA wal_checkpoint(TRUNCATE)")
                                
                                # Create new database and copy good data
                                with sqlite3.connect(temp_db) as new_conn:
                                    new_conn.execute("ATTACH DATABASE ? AS old_db", (db_file,))
                                    
                                    # Copy metadata
                                    new_conn.execute("CREATE TABLE IF NOT EXISTS metadata (run_id TEXT PRIMARY KEY, model TEXT, params TEXT)")
                                    new_conn.execute("INSERT OR IGNORE INTO metadata SELECT * FROM old_db.metadata")
                                    
                                    # Copy projections table structure
                                    new_conn.execute("""CREATE TABLE IF NOT EXISTS vec_projections (
                                        id INTEGER, run_id TEXT, model TEXT, mean_conf REAL, embedding BLOB,
                                        PRIMARY KEY(id, run_id)
                                    """)
                                    
                                    # Try to copy valid rows
                                    try:
                                        new_conn.execute("INSERT OR IGNORE INTO vec_projections SELECT * FROM old_db.vec_projections")
                                    except Exception:
                                        pass
                                    
                                    new_conn.commit()
                                    new_conn.execute("DETACH DATABASE old_db")
                                
                                temp_conn.close()
                                
                                # Rename corrupted DB and use recovered one
                                corrupted_db = f"{db_file}.corrupted"
                                if os.path.exists(corrupted_db):
                                    os.remove(corrupted_db)  # Remove old corrupted backup
                                os.rename(db_file, corrupted_db)
                                os.rename(temp_db, db_file)
                                print(f"[RECOVERY] Salvaged database, corrupted file renamed to {corrupted_db}", file=sys.stderr)
                                
                        except Exception as recovery_error:
                            print(f"[RECOVERY] Failed to recover {db_file}: {recovery_error}", file=sys.stderr)
                            # Fall back to renaming corrupted database (don't remove)
                            if os.path.exists(db_file):
                                corrupted_db = f"{db_file}.corrupted"
                                if os.path.exists(corrupted_db):
                                    os.remove(corrupted_db)
                                os.rename(db_file, corrupted_db)
                                print(f"[RECOVERY] Corrupted database renamed to {corrupted_db}", file=sys.stderr)
                    
                    time.sleep(1)  # Wait before retry
                    continue
            # If not a corruption error or retries exhausted, re-raise
            if retry == max_retries - 1:
                raise
            time.sleep(0.5)

# ---------- Load vectors ----------
def load_vectors_from_db(db_path, algo, run_id, ids=None) -> np.ndarray:
    db_file = os.path.join(db_path, f"{algo}.db")
    # Require vec0 for loading projection vectors to ensure vec_projections
    # is available and queries use the vec0 virtual table.
    conn = connect_vec_db(db_file, require_vec0=True)
    cur = conn.cursor()

    # Try to read from vec_projections; if virtual table exists we can select embedding directly
    try:
        if ids is None:
            cur.execute("SELECT id, embedding FROM vec_projections WHERE run_id=? ORDER BY id", (run_id,))
        else:
            placeholders = ",".join("?" * len(ids))
            cur.execute(f"SELECT id, embedding FROM vec_projections WHERE run_id=? AND id IN ({placeholders}) ORDER BY id", (run_id, *ids))
        rows = cur.fetchall()
    except Exception:
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
            try:
                conn.commit()
            except Exception:
                try:
                    conn.execute('COMMIT')
                except Exception:
                    pass

    if rows:
        cur.executemany("INSERT OR REPLACE INTO vec_metrics(img_id, model, embedding) VALUES (?, ?, ?)", rows)
        try:
            conn.commit()
        except Exception:
            try:
                conn.execute('COMMIT')
            except Exception:
                pass

    conn.close()

# ---------- kNN query example using vec0 ----------
def knn_query_projections(db_path, algo, query_vec: np.ndarray, k: int = 10, run_id: Optional[str] = None, min_conf: Optional[float] = None):
    db_file = os.path.join(db_path, f"{algo}.db")
    # Require vec0 for kNN queries; fail fast if vec0 is not available.
    conn = connect_vec_db(db_file, require_vec0=True)
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
    except Exception:
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
        # Build embeddings array and compute distances vectorized to avoid Python loops
        q = query_vec.astype(np.float32)
        blobs = [r[4] for r in all_rows]
        embeddings = np.stack([deserialize_float32_array(b, dim) for b in blobs], axis=0)
        # compute L2 distances in a vectorized manner
        dif = embeddings - q.reshape(1, -1)
        dists = np.linalg.norm(dif, axis=1)
        # get top-k indices
        topk_idx = np.argsort(dists)[:k]
        rows = []
        for idx in topk_idx:
            r = all_rows[idx]
            rows.append((r[0], r[1], r[2], r[3], float(dists[idx])))
    conn.close()
    return rows
