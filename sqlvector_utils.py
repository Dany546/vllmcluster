import sqlite3
import numpy as np
from typing import List, Tuple, Any

# Replace with the path to your sqlite vector extension binary if needed.
# If your Python build allows load_extension and the extension is installed in the system,
# you can set this to None and skip load_extension.
VECTOR_EXT_PATH = "/path/to/your/sqlite_vector_extension"  # <<-- set this

def connect_vec_db(db_path: str, load_extension: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    if load_extension and VECTOR_EXT_PATH:
        conn.enable_load_extension(True)
        conn.load_extension(VECTOR_EXT_PATH)
        conn.enable_load_extension(False)
    return conn

def create_embeddings_table(conn: sqlite3.Connection, dim: int, is_seg: bool = False):
    cur = conn.cursor()
    if is_seg:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                embedding VECTOR({dim}),
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
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                embedding VECTOR({dim}),
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
    cur.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()

def serialize_float32_array(arr: np.ndarray) -> bytes:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr.tobytes()

def insert_embeddings_batch(conn: sqlite3.Connection, rows: List[Tuple[Any, ...]]):
    cur = conn.cursor()
    ncols = len(rows[0])
    placeholders = ",".join(["?"] * ncols)
    cur.executemany(f"INSERT OR REPLACE INTO embeddings VALUES ({placeholders})", rows)
    conn.commit()

def build_vector_index(conn: sqlite3.Connection, table: str = "embeddings", column: str = "embedding"):
    cur = conn.cursor()
    # The exact index creation call depends on the extension.
    # Try common function names; adapt if your extension uses a different API.
    try:
        cur.execute(f"SELECT vector_create_index('{table}','{column}')")
    except sqlite3.OperationalError:
        try:
            cur.execute(f"SELECT vec_create_index('{table}','{column}')")
        except sqlite3.OperationalError:
            # Some extensions use CREATE INDEX ... USING hnsw(...) syntax
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_{column}_hnsw ON {table}({column})")
    conn.commit()

def query_knn(conn: sqlite3.Connection, query_vec: np.ndarray, k: int = 10):
    cur = conn.cursor()
    q_bytes = serialize_float32_array(query_vec)
    # Try operator '<->' first, fallback to function name
    try:
        cur.execute(f"SELECT id, img_id, mean_conf, box_loss, cls_loss, dfl_loss, flag_cat, flag_supercat, embedding <-> ? as dist FROM embeddings ORDER BY dist ASC LIMIT ?", (q_bytes, k))
    except sqlite3.OperationalError:
        cur.execute(f"SELECT id, img_id, mean_conf, box_loss, cls_loss, dfl_loss, flag_cat, flag_supercat, vector_distance(embedding, ?) as dist FROM embeddings ORDER BY dist ASC LIMIT ?", (q_bytes, k))
    return cur.fetchall()
