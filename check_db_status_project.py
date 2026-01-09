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
import hashlib

def compute_run_id(model: str, hyperparams: dict) -> str:
    """
    Compute a deterministic run_id from model name and hyperparameters (same algorithm as sqlvector_projector.compute_run_id)
    """
    hp_items = sorted(hyperparams.items())
    hp_str = json.dumps(hp_items)
    base = f"{model}:{hp_str}"
    run_hash = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{model}_{run_hash}"

# Defaults aligned with project.py
DEFAULT_EMBEDDINGS_DIR = "/CECI/home/ucl/irec/darimez/embeddings/"
DEFAULT_PROJ_DB_DIR = "/CECI/home/ucl/irec/darimez/proj"

# Hyperparameter grids (same as project.py)
umap_params = {
    "n_components": [2, 10, 25],
    "n_neighbors": [10, 20, 50, 100],
    "min_dist": [0.0, 0.1, 0.5], 
}
tsne_params = {
    "n_components": [2, 10, 25],
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
            # compute run_id using hyperparameters excluding the 'algo' key (matches project.py)
            run_id = compute_run_id(model_name, {k: v for k, v in params.items() if k != 'algo'})
            all_runs[("umap", run_id)] = (model_name, params)

    # t-SNE
    for model_name in model_names:
        for params in expand_params(tsne_params, "tsne"):
            run_id = compute_run_id(model_name, {k: v for k, v in params.items() if k != 'algo'})
            all_runs[("tsne", run_id)] = (model_name, params)

    return all_runs


def check_db(db_file, expected_run_ids, clean_partial=False, min_complete=5000, show_all=False, max_show=10, filter_expr=None):
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

    # If caller provided a filter expression, apply it to metadata.params (JSON stored in metadata.params)
    if filter_expr:
        from utils import parse_hyperparam_query, match_hyperparams

        # Build list of run_ids that match filter
        filtered = []
        for rr in metadata_rows:
            run_id = rr[0]
            try:
                cur.execute("SELECT params FROM metadata WHERE run_id=?", (run_id,))
                row = cur.fetchone()
                if row and row[0]:
                    params = json.loads(row[0])
                    if match_hyperparams(params, parse_hyperparam_query(filter_expr)):
                        filtered.append((run_id,))
            except Exception:
                # if params can't be parsed or don't exist, skip
                continue
        metadata_rows = filtered

        if not metadata_rows:
            print(f"No runs match filter: {filter_expr}")
    
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

    # Determine algorithm name from DB filename (e.g., tsne.db -> tsne)
    algo = os.path.splitext(os.path.basename(db_file))[0]

    # Compare expected runs (if provided) to metadata present in DB
    if expected_run_ids:
        # expected_run_ids keys are (algo, run_id)
        expected_for_algo = {run_id for (a, run_id) in expected_run_ids.keys() if a == algo}
        metadata_set = {r[0] for r in metadata_rows}
        missing_expected = sorted(expected_for_algo - metadata_set)
        unexpected = sorted(metadata_set - expected_for_algo)
        if missing_expected:
            print(f"{len(missing_expected)} expected runs are MISSING from metadata for algo {algo}:")
            iter_missing = missing_expected if show_all else missing_expected[:max_show]
            for rid in iter_missing:
                display_id = rid[:46] + ".." if len(rid) > 48 else rid
                model_name, params = expected_run_ids.get((algo, rid), (None, None))
                params_str = json.dumps(params, sort_keys=True) if params else ""
                print(f"  {display_id}  {model_name}  {params_str}")
            if not show_all and len(missing_expected) > max_show:
                print(f"  ... ({len(missing_expected)-max_show} more)")
            print("")
        if unexpected:
            print(f"{len(unexpected)} runs present in DB but not in expected grid for algo {algo}:")
            iter_unexp = unexpected if show_all else unexpected[:max_show]
            for rid in iter_unexp:
                display_id = rid[:46] + ".." if len(rid) > 48 else rid
                # try to fetch params from metadata if available
                params_str = ""
                model_name = ""
                try:
                    cur.execute("SELECT params, model FROM metadata WHERE run_id=?", (rid,))
                    row = cur.fetchone()
                    if row and row[0]:
                        try:
                            params = json.loads(row[0])
                            params_str = json.dumps(params, sort_keys=True)
                            model_name = row[1] or ""
                        except Exception:
                            params_str = str(row[0])
                    else:
                        # fallback: infer params from columns other than run_id and model
                        cur.execute("PRAGMA table_info(metadata)")
                        cols = [r[1] for r in cur.fetchall()]
                        other_cols = [c for c in cols if c not in ("run_id","model")]
                        if other_cols:
                            sel = ", ".join(other_cols)
                            cur.execute(f"SELECT {sel} FROM metadata WHERE run_id=?", (rid,))
                            vals = cur.fetchone()
                            if vals:
                                params = dict(zip(other_cols, vals))
                                params_str = json.dumps(params, sort_keys=True)
                except Exception:
                    pass
                print(f"  {display_id}  {model_name}  {params_str}")
            if not show_all and len(unexpected) > max_show:
                print(f"  ... ({len(unexpected)-max_show} more)")
            print("")

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
    parser.add_argument("--filter", default=None, help="Filter runs by hyperparameters, e.g. 'n_neighbors=20,min_dist>0.1'")
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
    tables = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith(".db") and not f=="attention.db"]
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
        check_db(db_file, expected, clean_partial=args.clean, show_all=show_all, max_show=max_show, filter_expr=args.filter)

    print("\nSummary complete.")

if __name__ == "__main__":
    main()
