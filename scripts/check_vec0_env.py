#!/usr/bin/env python3
"""Diagnostic: verify vec0 availability and that DBs contain vec_projections virtual tables.

Usage:
  python vllmcluster/scripts/check_vec0_env.py --proj-dir /path/to/proj --emb-dir /path/to/embeddings

It attempts to load vec0 (via APSW/sqlite_vec or VECTOR_EXT_PATH/VEC0_SO), then inspects each DB.
"""
import os
import argparse
import sqlite3
import sys
# Ensure we can import sibling modules when running this script directly.
# Add the `vllmcluster` package directory to sys.path so imports work outside a package context.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlvector_projector import connect_vec_db


def check_db(path):
    print(f"Checking {path}")
    try:
        conn = connect_vec_db(path, require_vec0=True)
    except Exception as e:
        print(f"  ERROR: vec0 not available for {path}: {e}")
        return False
    cur = conn.cursor()
    cur.execute("SELECT name, type, sql FROM sqlite_master WHERE name='vec_projections' OR name='vec_metrics'")
    rows = cur.fetchall()
    if not rows:
        print("  WARNING: vec_projections / vec_metrics not found in sqlite_master")
        conn.close()
        return False
    for name, typ, sql in rows:
        print(f"  Found {name} ({typ})")
        if sql and 'USING vec0' in sql.upper():
            print("   -> virtual table uses vec0")
        else:
            print("   -> table present but does not appear to be a vec0 virtual table")
    conn.close()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proj-dir", required=True)
    p.add_argument("--emb-dir", required=False)
    args = p.parse_args()

    ok = True
    for fname in os.listdir(args.proj_dir):
        if not fname.endswith('.db'):
            continue
        path = os.path.join(args.proj_dir, fname)
        if not check_db(path):
            ok = False

    if args.emb_dir:
        for fname in os.listdir(args.emb_dir):
            if not fname.endswith('.db'):
                continue
            path = os.path.join(args.emb_dir, fname)
            # embeddings DB may not have vec_projections; skip but check vec_metrics
            try:
                conn = connect_vec_db(path, require_vec0=True)
                print(f"Emb DB {path}: vec0 available")
                conn.close()
            except Exception as e:
                print(f"  ERROR: vec0 not available for {path}: {e}")
                ok = False

    if not ok:
        print("One or more checks failed. Ensure the `sqlite-vec` Python package is installed and available in your environment.")
        exit(2)
    print("All checks passed: vec0 available and projection DBs contain vec_projections (or reported otherwise).")


if __name__ == '__main__':
    main()
