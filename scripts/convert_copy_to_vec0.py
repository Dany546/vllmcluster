#!/usr/bin/env python3
"""
Create vec0 virtual tables in an existing DB copy and populate them from the fallback table.

Usage:
  python scripts/convert_copy_to_vec0.py --db /path/to/umap.vec0.db --run-id <run_id>

The script is non-destructive: it creates new virtual tables named `vec_projections_v` and
`vec_metrics_v` (if applicable) and populates them from existing `vec_projections` / `vec_metrics`.
After verification you may rename/drop tables or replace the DB file.
"""
import argparse
import os
import sqlite3
import numpy as np

try:
    import apsw
except Exception:
    apsw = None

try:
    import sqlite_vec
except Exception:
    sqlite_vec = None


def find_vecso():
    # prefer explicit env var
    # prefer modern VECTOR_EXT_PATH but accept legacy VEC0_SO
    if "VECTOR_EXT_PATH" in os.environ:
        return os.environ.get("VECTOR_EXT_PATH")
    if "VEC0_SO" in os.environ:
        return os.environ.get("VEC0_SO")
    if sqlite_vec is not None:
        base = os.path.dirname(getattr(sqlite_vec, "__file__", ""))
        cand = os.path.join(base, "vec0.so")
        if os.path.exists(cand):
            return cand
    return None


def get_dim_from_blob(blob):
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.size


def create_and_populate(db_path: str, run_id: str):
    if apsw is None:
        raise RuntimeError("APSW is required to load vec0 and create virtual tables in this script.")

    vecso = find_vecso()
    if vecso is None:
        print("Warning: could not locate vec0.so via sqlite-vec installation; ensure the `sqlite-vec` Python package is installed in this environment.")

    # fetch a sample blob to infer dim using sqlite3 (safe read)
    sconn = sqlite3.connect(db_path)
    scur = sconn.cursor()
    scur.execute("SELECT embedding FROM vec_projections WHERE run_id=? LIMIT 1", (run_id,))
    row = scur.fetchone()
    if not row:
        sconn.close()
        raise RuntimeError(f"No rows found for run_id={run_id}")
    sample_blob = row[0]
    dim = get_dim_from_blob(sample_blob)
    sconn.close()

    # use APSW connection to create virtual table; sqlite-vec must be installed in the environment
    aconn = apsw.Connection(db_path)
    try:
        import sqlite_vec
        try:
            sqlite_vec.load(aconn)
            print('Registered vec0 via sqlite-vec')
        except Exception as e:
            print('Failed to register vec0 via sqlite-vec:', e)
    except Exception:
        print('sqlite-vec package not importable in this environment; ensure it is installed')

    cur = aconn.cursor()

    # create virtual table name
    vt_name = 'vec_projections_v'
    try:
        cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {vt_name} USING vec0(id UNINDEXED, run_id UNINDEXED, model UNINDEXED, mean_conf, embedding FLOAT[{dim}])")
        print('Created virtual table', vt_name)
    except Exception as e:
        print('Could not create vec0 virtual table:', e)
        print('Aborting conversion for', db_path)
        return

    # populate from fallback table
    try:
        cur.execute(f"INSERT OR REPLACE INTO {vt_name}(id, run_id, model, mean_conf, embedding) SELECT id, run_id, model, mean_conf, embedding FROM vec_projections WHERE run_id=?", (run_id,))
        print('Populated virtual table from vec_projections for run', run_id)
    except Exception as e:
        print('Failed to populate virtual table:', e)

    # verify counts
    try:
        src = cur.execute("SELECT count(*) FROM vec_projections WHERE run_id=?", (run_id,)).fetchone()[0]
        dst = cur.execute(f"SELECT count(*) FROM {vt_name} WHERE run_id=?", (run_id,)).fetchone()[0]
        print(f"Counts for run {run_id}: source={src}, vec0={dst}")
    except Exception as e:
        print('Verification query failed:', e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()

    create_and_populate(args.db, args.run_id)


if __name__ == '__main__':
    main()
