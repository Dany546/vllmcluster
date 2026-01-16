# sqlvector_utils.py
import numpy as np
from typing import List, Tuple, Any, Optional
import os

try:
    import apsw
except Exception:
    apsw = None

# Type alias for connection to work with both APSW and sqlite3
Connection = Any

def connect_vec_db(db_path: str, require_vec0: bool = False) -> Connection:
    """Connect to SQLite using APSW and load sqlite-vec extension if available.

    If `require_vec0` is True this function raises if vec0 can't be loaded.
    """
    if apsw is None:
        raise RuntimeError('APSW is required. Install via `pip install apsw`')
    
    # Use APSW as the primary method for extension loading
    try:
        aconn = apsw.Connection(db_path)
        
        # Set PRAGMA settings for performance
        try:
            aconn.execute("PRAGMA journal_mode=WAL")
            aconn.execute("PRAGMA synchronous=NORMAL")
            aconn.execute("PRAGMA temp_store=MEMORY")
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

def create_embeddings_table(conn: Connection, dim: int, is_seg: bool = False):
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
    # Commit for both sqlite3 and APSW connections
    try:
        conn.commit()
    except Exception:
        try:
            conn.execute('COMMIT')
        except Exception:
            pass

def serialize_float32_array(arr: np.ndarray) -> bytes:
    """Serialize numpy float32 array to bytes compatible with vec0 BLOB format."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # Use numpy's tobytes() method for serialization
    return arr.tobytes()

def deserialize_float32_array(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes back to numpy float32 array."""
    # Accept either a numpy array (if the connection returned one) or raw bytes
    if isinstance(blob, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(blob, dtype=np.float32)
    else:
        # assume DB returned an array-like object
        arr = np.array(blob, dtype=np.float32)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} dimensions, got {arr.size}")
    return arr

def insert_embeddings_batch(conn: Connection, rows: List[Tuple[Any, ...]]):
    """Insert or replace a batch of rows into embeddings table."""
    if not rows:
        return
    cur = conn.cursor()
    ncols = len(rows[0])
    placeholders = ",".join(["?"] * ncols)
    cur.executemany(f"INSERT OR REPLACE INTO embeddings VALUES ({placeholders})", rows)
    # Commit for both sqlite3 and APSW connections
    try:
        conn.commit()
    except Exception:
        try:
            conn.execute('COMMIT')
        except Exception:
            pass

def build_vector_index(conn: Connection, table: str = "embeddings", column: str = "embedding", metric: str = "l2"):
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
        # Commit for both sqlite3 and APSW connections
        try:
            conn.commit()
        except Exception:
            try:
                conn.execute('COMMIT')
            except Exception:
                pass
    except Exception:
        # vec0 not available; leave as-is and rely on linear scan
        pass

def query_knn(
    conn: Connection,
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
    except Exception:
        # fallback to linear scan
        return query_knn_linear(conn, query_vec, k, table, return_distances, filter_clause, filter_params)

def query_knn_linear(
    conn: Connection,
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

def get_embedding_by_id(conn: Connection, emb_id: int) -> Optional[np.ndarray]:
    """Retrieve embedding vector by ID. Returns numpy float32 array or None."""
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM embeddings WHERE id = ?", (emb_id,))
    row = cur.fetchone()
    if row is None:
        return None
    dim = int(cur.execute("SELECT value FROM metadata WHERE key = 'dimension'").fetchone()[0])
    return deserialize_float32_array(row[0], dim)

def get_all_embeddings(conn: Connection) -> Tuple[np.ndarray, List[int]]:
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
