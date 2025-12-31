# sqlvector_utils.py
import sqlite3
import numpy as np
from typing import List, Tuple, Any, Optional
import os

# Optional helper package that registers vec0 and helper functions
try:
    import sqlite_vec
except Exception:
    sqlite_vec = None

VECTOR_EXT_PATH = os.getenv("VECTOR_EXT_PATH", None)

def connect_vec_db(db_path: str) -> sqlite3.Connection:
    """Connect to SQLite and try to load sqlite-vec if available."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-20000")
    # Try to register sqlite-vec via its Python loader if available
    if sqlite_vec is not None:
        try:
            sqlite_vec.load(conn)
            # verify extension loaded (vec_version returns a single value)
            conn.execute("SELECT vec_version()").fetchone()
        except Exception:
            # If load fails, continue; functions will fall back to linear scan
            pass
    else:
        # If sqlite_vec not installed but VECTOR_EXT_PATH points to a library, try to load it
        if VECTOR_EXT_PATH and os.path.exists(VECTOR_EXT_PATH):
            try:
                conn.enable_load_extension(True)
                conn.load_extension(VECTOR_EXT_PATH)
                conn.enable_load_extension(False)
                conn.execute("SELECT vec_version()").fetchone()
            except Exception:
                # ignore and fall back
                pass
    return conn

def create_embeddings_table(conn: sqlite3.Connection, dim: int, is_seg: bool = False):
    """Create embeddings and metadata tables."""
    cur = conn.cursor()
    if is_seg:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                embedding BLOB,
                hit_freq REAL,
                mean_iou REAL,
                mean_conf REAL,
                mean_dice REAL,
                flag_cat INTEGER,
                flag_supercat INTEGER,
                box_loss REAL,
                cls_loss REAL,
                dfl_loss REAL,
                seg_loss REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                iou BLOB,
                conf BLOB,
                dice BLOB,
                cat BLOB,
                supercat BLOB
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                embedding BLOB,
                hit_freq REAL,
                mean_iou REAL,
                mean_conf REAL,
                flag_cat INTEGER,
                flag_supercat INTEGER,
                box_loss REAL,
                cls_loss REAL,
                dfl_loss REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                iou BLOB,
                conf BLOB,
                cat BLOB,
                supercat BLOB
            )
        """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cur.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", ("dimension", str(dim)))
    cur.execute("CREATE INDEX IF NOT EXISTS idx_img_id ON embeddings(img_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_flag_cat ON embeddings(flag_cat)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_flag_supercat ON embeddings(flag_supercat)")
    conn.commit()

def serialize_float32_array(arr: np.ndarray) -> bytes:
    """Serialize numpy float32 array to bytes compatible with vec0 BLOB format."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # prefer sqlite_vec helper if present
    if sqlite_vec is not None and hasattr(sqlite_vec, "serialize_float32"):
        try:
            # sqlite_vec.serialize_float32 expects a Python list or similar
            return sqlite_vec.serialize_float32(arr.tolist())
        except Exception:
            pass
    return arr.tobytes()

def deserialize_float32_array(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes back to numpy float32 array."""
    if sqlite_vec is not None and hasattr(sqlite_vec, "deserialize_float32"):
        try:
            arr = np.array(sqlite_vec.deserialize_float32(blob), dtype=np.float32)
            if arr.size != dim:
                raise ValueError(f"Expected {dim} dimensions, got {arr.size}")
            return arr
        except Exception:
            pass
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} dimensions, got {arr.size}")
    return arr

def insert_embeddings_batch(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]]):
    """Insert or replace a batch of rows into embeddings table."""
    if not rows:
        return
    cur = conn.cursor()
    ncols = len(rows[0])
    placeholders = ",".join(["?"] * ncols)
    cur.executemany(f"INSERT OR REPLACE INTO embeddings VALUES ({placeholders})", rows)
    conn.commit()

def build_vector_index(conn: sqlite3.Connection, table: str = "embeddings", column: str = "embedding", metric: str = "l2"):
    """Create vec0 virtual table and populate it. Falls back silently if vec0 unavailable."""
    cur = conn.cursor()
    # read dimension from metadata
    row = cur.execute("SELECT value FROM metadata WHERE key = 'dimension'").fetchone()
    if row is None:
        raise RuntimeError("dimension not set in metadata")
    dim = int(row[0])
    try:
        cur.execute(f"DROP TABLE IF EXISTS vec_{table}")
        cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{table} USING vec0(id UNINDEXED, embedding FLOAT[{dim}] distance_metric={metric})")
        cur.execute(f"INSERT INTO vec_{table} (id, embedding) SELECT id, {column} FROM {table}")
        conn.commit()
    except sqlite3.OperationalError:
        # vec0 not available; leave as-is and rely on linear scan
        pass

def query_knn(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int = 10,
    table: str = "embeddings",
    return_distances: bool = True,
    filter_clause: Optional[str] = None,
    filter_params: Optional[tuple] = None
) -> List[Tuple]:
    """kNN query using vec0 if available, otherwise linear scan."""
    cur = conn.cursor()
    q_bytes = serialize_float32_array(query_vec)
    try:
        if filter_clause:
            query = f"""
                SELECT e.id, e.img_id, e.mean_conf, e.box_loss, e.cls_loss, e.dfl_loss,
                       e.flag_cat, e.flag_supercat, vec_distance_L2(v.embedding, ?) as dist
                FROM vec_{table} v
                JOIN {table} e ON v.id = e.id
                WHERE {filter_clause}
                ORDER BY dist
                LIMIT ?
            """
            params = (q_bytes, *filter_params, k) if filter_params else (q_bytes, k)
        else:
            query = f"""
                SELECT e.id, e.img_id, e.mean_conf, e.box_loss, e.cls_loss, e.dfl_loss,
                       e.flag_cat, e.flag_supercat, vec_distance_L2(v.embedding, ?) as dist
                FROM vec_{table} v
                JOIN {table} e ON v.id = e.id
                ORDER BY dist
                LIMIT ?
            """
            params = (q_bytes, k)
        cur.execute(query, params)
        rows = cur.fetchall()
        if not return_distances:
            rows = [r[:-1] for r in rows]
        return rows
    except sqlite3.OperationalError:
        # fallback to linear scan
        return query_knn_linear(conn, query_vec, k, table, return_distances, filter_clause, filter_params)

def query_knn_linear(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int = 10,
    table: str = "embeddings",
    return_distances: bool = True,
    filter_clause: Optional[str] = None,
    filter_params: Optional[tuple] = None
) -> List[Tuple]:
    """Linear scan kNN fallback."""
    cur = conn.cursor()
    if filter_clause:
        cur.execute(f"SELECT id, img_id, embedding, mean_conf, box_loss, cls_loss, dfl_loss, flag_cat, flag_supercat FROM {table} WHERE {filter_clause}", filter_params or ())
    else:
        cur.execute(f"SELECT id, img_id, embedding, mean_conf, box_loss, cls_loss, dfl_loss, flag_cat, flag_supercat FROM {table}")
    rows = cur.fetchall()
    if not rows:
        return []
    dim = int(conn.execute("SELECT value FROM metadata WHERE key = 'dimension'").fetchone()[0])
    distances = []
    for row in rows:
        emb = deserialize_float32_array(row[2], dim)
        dist = float(np.linalg.norm(query_vec.astype(np.float32) - emb))
        distances.append((dist, row))
    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    results = []
    for dist, row in top_k:
        if return_distances:
            results.append((*row[:2], *row[3:], dist))
        else:
            results.append((*row[:2], *row[3:]))
    return results

def get_embedding_by_id(conn: sqlite3.Connection, emb_id: int) -> Optional[np.ndarray]:
    """Retrieve embedding vector by ID. Returns numpy float32 array or None."""
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM embeddings WHERE id = ?", (emb_id,))
    row = cur.fetchone()
    if row is None:
        return None
    dim = int(cur.execute("SELECT value FROM metadata WHERE key = 'dimension'").fetchone()[0])
    return deserialize_float32_array(row[0], dim)

def get_all_embeddings(conn: sqlite3.Connection) -> Tuple[np.ndarray, List[int]]:
    """Return (embeddings_array, ids). embeddings_array shape is (N, dim)."""
    cur = conn.cursor()
    cur.execute("SELECT id, embedding FROM embeddings ORDER BY id")
    rows = cur.fetchall()
    if not rows:
        return np.array([]), []
    dim = int(cur.execute("SELECT value FROM metadata WHERE key = 'dimension'").fetchone()[0])
    ids = []
    embeddings = []
    for row_id, emb_blob in rows:
        ids.append(row_id)
        embeddings.append(deserialize_float32_array(emb_blob, dim))
    return np.vstack(embeddings), ids
