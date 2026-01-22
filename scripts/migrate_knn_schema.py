#!/usr/bin/env python3
"""
Small migration utility to add sanitized columns and copy values from legacy metric columns.
Usage: python scripts/migrate_knn_schema.py /path/to/knn_results.db
If no path is provided, uses the module's DB_PATH.
"""
import sqlite3
import sys
from pathlib import Path

from vllmcluster import evaluate_clusters as evalc


def migrate(db_path=None):
    if db_path is None:
        db_path = evalc.DB_PATH
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("PRAGMA table_info(knn_results)")
    cols = [r[1] for r in c.fetchall()]

    # Map legacy->sanitized
    if "mae/accuracy" in cols and "mae" not in cols:
        c.execute('ALTER TABLE "knn_results" ADD COLUMN "mae" REAL')
        c.execute('UPDATE "knn_results" SET "mae" = "mae/accuracy" WHERE "mae/accuracy" IS NOT NULL')
        print('Migrated mae/accuracy -> mae')

    if "correlation/ARI" in cols and "corr" not in cols:
        c.execute('ALTER TABLE "knn_results" ADD COLUMN "corr" REAL')
        c.execute('UPDATE "knn_results" SET "corr" = "correlation/ARI" WHERE "correlation/ARI" IS NOT NULL')
        print('Migrated correlation/ARI -> corr')

    conn.commit()
    conn.close()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    migrate(arg)
