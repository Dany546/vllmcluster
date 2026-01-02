#!/usr/bin/env python3
"""
Non-destructive migration of a proj DB to a vec0-backed copy and micro-benchmarking.

Usage:
  python scripts/migrate_proj_to_vec0.py --src /path/to/umap.db --algo umap --out /path/to/umap.vec0.db --run-id <run_id> --queries 100

This script will:
 - copy the source DB to the `--out` path
 - attempt to load vec0 (via APSW/sqlvector_projector.connect_vec_db)
 - create vec tables in the destination and insert all rows for the given `run_id`
 - run repeated k-NN queries and time them (compare to linear fallback if vec0 not available)

Make sure your venv has `apsw` installed or `VEC0_SO` env var points to a vec0 .so file.
"""
import argparse
import os
import shutil
import time
import sqlite3
from typing import List

import numpy as np

# Import helpers from the package
from sqlvector_projector import (
    connect_vec_db,
    create_vec_tables,
    insert_projection_batch,
    load_vectors_from_db,
    knn_query_projections,
    serialize_float32_array,
)


def copy_db(src: str, dst: str):
    if os.path.exists(dst):
        raise FileExistsError(f"Destination exists: {dst}")
    shutil.copy2(src, dst)


def read_runs_from_db(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT run_id FROM vec_projections")
        rows = cur.fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        # maybe legacy table with run_id column
        try:
            cur.execute("SELECT DISTINCT run_id FROM vec_projections")
            return [r[0] for r in cur.fetchall()]
        except Exception:
            return []
    finally:
        conn.close()


def migrate_run(src_db: str, dst_db: str, algo: str, run_id: str, batch_size: int = 512):
    # destination: open with connect_vec_db to ensure vec0 is loaded if possible
    dst_conn = connect_vec_db(dst_db)
    # create vec tables with unknown metrics dim (0)
    create_vec_tables(dst_conn, proj_dim=2, metrics_dim=0)

    # read rows from source using sqlite3 (safe)
    sconn = sqlite3.connect(src_db)
    scur = sconn.cursor()
    scur.execute("PRAGMA table_info(vec_projections)")
    cols = [r[1] for r in scur.fetchall()]

    if 'embedding' not in cols:
        raise RuntimeError("Source DB does not have vec_projections.embedding column")

    scur.execute("SELECT id, run_id, model, mean_conf, embedding FROM vec_projections WHERE run_id=? ORDER BY id", (run_id,))

    batch = []
    count = 0
    for row in scur:
        _id, r_id, model, mean_conf, blob = row
        # convert blob to numpy then back to serialized form to ensure compatibility
        vec = np.frombuffer(blob, dtype=np.float32)
        batch.append((_id, r_id, model, mean_conf if mean_conf is not None else 0.0, vec))
        if len(batch) >= batch_size:
            insert_projection_batch(dst_conn, batch, proj_dim=vec.size)
            count += len(batch)
            print(f"Inserted {count} rows...")
            batch = []
    if batch:
        insert_projection_batch(dst_conn, batch, proj_dim=batch[0][4].size)
        count += len(batch)
        print(f"Inserted {count} rows (final)")

    sconn.close()
    try:
        dst_conn.close()
    except Exception:
        pass


def benchmark_knn(db_path: str, algo: str, run_id: str, queries: int = 100, k: int = 10):
    # pick a random sample embedding to query
    vectors = load_vectors_from_db(os.path.dirname(db_path), algo, run_id)
    if vectors.size == 0:
        print("No vectors found; skipping benchmark")
        return
    rng = np.random.default_rng(42)
    idxs = rng.integers(0, vectors.shape[0], size=min(queries, vectors.shape[0]))
    times = []
    for i in range(len(idxs)):
        q = vectors[idxs[i]]
        t0 = time.perf_counter()
        _ = knn_query_projections(os.path.dirname(db_path), algo, q, k=k, run_id=run_id)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    print(f"Ran {len(times)} queries: mean={times.mean():.6f}s median={np.median(times):.6f}s p95={np.percentile(times,95):.6f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source proj DB file path (e.g. /.../umap.db)")
    p.add_argument("--algo", choices=["umap","tsne"], default="umap")
    p.add_argument("--out", required=True, help="Destination DB path for vec0 copy")
    p.add_argument("--run-id", required=False, help="Specific run_id to migrate; if omitted the script will pick the first available run")
    p.add_argument("--queries", type=int, default=200, help="Number of random k-NN queries to run for benchmarking")
    p.add_argument("--k", type=int, default=10, help="k for k-NN queries")
    args = p.parse_args()

    print(f"Copying {args.src} -> {args.out} (non-destructive)")
    copy_db(args.src, args.out)

    # find a run_id to migrate if not provided
    if not args.run_id:
        runs = read_runs_from_db(args.src)
        if not runs:
            raise RuntimeError("No run_id found in source DB; pass --run-id explicitly")
        args.run_id = runs[0]
        print(f"Auto-selected run_id: {args.run_id}")

    print("Starting migration... this may take a while for large DBs")
    migrate_run(args.src, args.out, args.algo, args.run_id)

    print("Migration completed. Running benchmark against destination DB")
    benchmark_knn(args.out, args.algo, args.run_id, queries=args.queries, k=args.k)


if __name__ == '__main__':
    main()
