#!/usr/bin/env python3
"""
Check the status of projection databases produced by project.py

Usage:
    python check_db_status_project.py --proj-dir PROJ_DIR --emb-dir EMB_DIR [--clean] [--algos tsne,umap]

This script inspects `metadata` and `embeddings` tables (or falls back to `vec_projections`) and
reports COMPLETE (≥5000), PARTIAL (1-4999) and EMPTY runs for each model and algorithm.
"""
import os
import sys
import argparse
import sqlite3
import json

# Add project root to path so imports work if invoked from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sqlvector_projector import compute_run_id

# Defaults aligned with project.py
DEFAULT_EMBEDDINGS_DIR = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
DEFAULT_PROJ_DB_DIR = "/globalscratch/ucl/irec/darimez/dino/proj2/"

# Hyperparameter grids (same as project.py)
umap_params = {
    "n_components": [2, 10, 25],
    "n_neighbors": [10, 20, 50, 100],
    "min_dist": [0.0, 0.1, 0.5],
}
tsne_params = {
    "n_components": [10, 25],
    "perplexity": [40],
    "early_exaggeration": [20],
    "learning_rate": [200],
    "max_iter": [1000],
}


def expand_params(param_dict, algo):
    keys = list(param_dict.keys())
    for values in __import__("itertools").product(*[param_dict[k] for k in keys]):
        d = dict(zip(keys, values))
        d["algo"] = algo
        yield d


def gather_expected_runs(model_names):
    """Return dict mapping (algo, run_id) -> hyperparams for all models"""
    all_runs = {}
    # UMAP
    for model_name in model_names:
        for params in expand_params(umap_params, "umap"):
            run_id = compute_run_id(model_name, params)
            all_runs[("umap", run_id)] = (model_name, params)

    # t-SNE
    for model_name in model_names:
        for params in expand_params(tsne_params, "tsne"):
            run_id = compute_run_id(model_name, params)
            all_runs[("tsne", run_id)] = (model_name, params)

    return all_runs


def check_db(db_file, expected_run_ids, clean_partial=False, min_complete=5000, show_all=False, max_show=10):
    if not os.path.exists(db_file):
        print(f"Database file does not exist: {db_file}")
        return None

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # get metadata runs
    try:
        cur.execute("SELECT run_id, json_remove(json_type(params), '$') FROM metadata")
    except Exception:
        # metadata may not use params; fallback to simple query
        try:
            cur.execute("SELECT run_id, model FROM metadata")
        except Exception:
            # give up
            pass

    # Fetch run ids from metadata robustly (support multiple schema variants)
    metadata_rows = []
    try:
        cur.execute("SELECT run_id FROM metadata")
        metadata_rows = [(r[0],) for r in cur.fetchall()]
    except Exception:
        metadata_rows = []

    if not metadata_rows:
        print("No runs found in database metadata.")
    
    # counts: prefer embeddings table (project.py) then vec_projections (sqlvector)
    count_rows = {}
    try:
        cur.execute("SELECT run_id, COUNT(*) as cnt FROM embeddings GROUP BY run_id")
        for row in cur.fetchall():
            count_rows[row[0]] = row[1]
    except Exception:
        try:
            cur.execute("SELECT run_id, COUNT(*) as cnt FROM vec_projections GROUP BY run_id")
            for row in cur.fetchall():
                count_rows[row[0]] = row[1]
        except Exception:
            pass

    # Summarize
    # Classify runs
    complete_list = []
    partial_list = []  # (run_id, count)
    empty_list = []
    for run_row in metadata_rows:
        run_id = run_row[0]
        count = count_rows.get(run_id, 0)
        if count >= min_complete:
            complete_list.append(run_id)
        elif count > 0:
            partial_list.append((run_id, count))
        else:
            empty_list.append(run_id)

    total_runs = len(metadata_rows)

    print(f"\nDatabase: {os.path.basename(db_file)}")
    print("-" * 80)
    # Show partial runs (important to surface)
    if partial_list:
        print("Partial runs (need attention):")
        for run_id, cnt in partial_list:
            display_id = run_id[:46] + ".." if len(run_id) > 48 else run_id
            print(f"  {display_id.ljust(48)} {cnt} rows")
        print("")

    # Show empty runs
    if empty_list:
        print("Empty runs:")
        for run_id in empty_list:
            display_id = run_id[:46] + ".." if len(run_id) > 48 else run_id
            print(f"  {display_id}")
        print("")

    # Show a sample of complete runs unless caller asked for all
    if show_all:
        print("Complete runs:")
        for run_id in complete_list:
            display_id = run_id[:46] + ".." if len(run_id) > 48 else run_id
            print(f"  {display_id}")
    else:
        nshow = min(len(complete_list), max_show)
        if nshow > 0:
            print(f"Showing {nshow} of {len(complete_list)} complete runs (use --show-all to list all):")
            for run_id in complete_list[:nshow]:
                display_id = run_id[:46] + ".." if len(run_id) > 48 else run_id
                print(f"  {display_id}")
        else:
            print("No complete runs to show.")

    print("-" * 80)
    print(f"Total runs in DB: {total_runs}")
    print(f"Complete (≥{min_complete}): {len(complete_list)}")
    print(f"Partial (1-{min_complete-1}): {len(partial_list)}")

    # Optionally clean partial runs
    if clean_partial and partial_list:
        partial_run_ids = [run_id for run_id, cnt in partial_list]
        print(f"\nRemoving {len(partial_run_ids)} partial runs...")
        for r in partial_run_ids:
            try:
                cur.execute("DELETE FROM embeddings WHERE run_id = ?", (r,))
                cur.execute("DELETE FROM metadata WHERE run_id = ?", (r,))
                print(f"  Removed: {r}")
            except Exception as e:
                print(f"  Failed to remove {r}: {e}")
        try:
            conn.commit()
        except Exception:
            try:
                conn.execute("COMMIT")
            except Exception:
                pass

    conn.close()
    return {
        "total": total_runs,
        "complete": len(complete_list),
        "partial": len(partial_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Check DB status for project.py outputs")
    parser.add_argument("--proj-dir", default=DEFAULT_PROJ_DB_DIR, help="Directory containing projection DBs (tsne.db, umap.db)")
    parser.add_argument("--emb-dir", default=DEFAULT_EMBEDDINGS_DIR, help="Directory containing embedding .db files (per-model)")
    parser.add_argument("--clean", action="store_true", help="Remove partial runs after showing status")
    parser.add_argument("--algos", default="tsne,umap", help="Comma-separated algorithms to check (e.g., tsne,umap)")
    parser.add_argument("--show-all", action="store_true", help="Show full list of complete runs instead of a sample")
    parser.add_argument("--max-show", type=int, default=10, help="Max number of complete runs to display when not using --show-all")
    args = parser.parse_args()

    emb_dir = args.emb_dir
    proj_dir = args.proj_dir
    show_all = args.show_all
    max_show = args.max_show

    if not os.path.exists(emb_dir):
        print(f"Embeddings dir does not exist: {emb_dir}")
        return

    # Gather model names from embedding DBs
    tables = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith(".db")]
    if not tables:
        print("No embedding .db files found in embedding dir")
        return

    model_names = [os.path.basename(t).split(".")[0] for t in tables]
    print(f"Found models: {', '.join(model_names)}")

    expected = gather_expected_runs(model_names)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for algo in algos:
        db_file = os.path.join(proj_dir, f"{algo}.db")
        print("\n" + "=" * 70)
        print(f"Checking algorithm: {algo.upper()} -> {db_file}")
        print("=" * 70)
        check_db(db_file, expected, clean_partial=args.clean, show_all=show_all, max_show=max_show)

    print("\nSummary complete.")

if __name__ == "__main__":
    main()
