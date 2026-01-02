"""
Project helpers for sql-vector projection DBs.

These helpers provide a compatible view for code that previously
enumerated projection runs via plain `embeddings` tables. They are
intended to be imported by new, sql-vector-aware pipeline entrypoints
or used to incrementally port existing code.
"""
import os
import sqlite3
from typing import List, Tuple


def list_proj_runs(proj_dir: str) -> Tuple[List[str], List[str]]:
    """Enumerate projection DBs and their run identifiers.

    Returns (table_names, table_paths) where `table_names` are strings in
    the legacy format `model.algo.hash` and `table_paths` are the
    corresponding .db file paths (one entry per run).
    """
    lower_dim_tables = [
        os.path.join(proj_dir, file)
        for file in os.listdir(proj_dir)
        if file.endswith(".db") and file.split(".")[0] not in ["metrics"]
    ]

    table_names = []
    table_paths = []
    for db in lower_dim_tables:
        algo = os.path.basename(db).split(".")[0]
        conn = sqlite3.connect(db)
        c = conn.cursor()
        # prefer metadata table if present
        run_ids = []
        try:
            c.execute("SELECT run_id FROM metadata")
            run_ids = [r[0] for r in c.fetchall()]
        except Exception:
            try:
                c.execute("SELECT DISTINCT run_id FROM vec_projections")
                run_ids = [r[0] for r in c.fetchall()]
            except Exception:
                run_ids = []
        conn.close()

        for run_id in run_ids:
            if "_" in run_id:
                model, hashpart = run_id.split("_", 1)
                table_names.append(f"{model}.{algo}.{hashpart}")
            else:
                table_names.append(f"{run_id}.{algo}.")
            table_paths.append(db)

    return table_names, table_paths
