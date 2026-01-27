"""Orchestrator for running the gridsearch.

This script reads a YAML config, builds pipeline configurations, and dispatches
jobs via joblib to `worker_run` from `vllmcluster.jobs`.

This is a minimal implementation to start; embedding loading and distances
handling should be provided by the caller or extended in a follow-up.
"""
from __future__ import annotations

import argparse
import os
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import numpy as np
import sqlite3

import db_utils
import json
import hashlib
import bad_runs
from jobs import worker_run


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_knn_grid():
    n_list = [5, 10, 20, 30, 50]
    metrics = ['euclidean', 'cosine']
    grid = []
    for n in n_list:
        for m in metrics:
            grid.append({'n_neighbors': n, 'metric': m})
    return grid


def build_knn_grid_distances():
    """For precomputed distances, only vary n_neighbors (metric='precomputed')."""
    n_list = [5, 10, 20, 30, 50]
    grid = []
    for n in n_list:
        grid.append({'n_neighbors': n, 'metric': 'precomputed'})
    return grid


def load_and_symmetrize_distances(distances_db_path, target: str | None = None, expected_ids: list | None = None):
    """Load distances from DB and symmetrize by taking mean of dij and dji.

    If `target` corresponds to a component (e.g., 'seg_loss' -> 'seg'), prefer
    that component matrix when available. Falls back to 'total' or the first
    available component.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Before reading the matrix, check coverage of IDs if expected_ids is provided
    try:
        if expected_ids is not None:
            conn = sqlite3.connect(distances_db_path)
            cur = conn.cursor()
            try:
                cur.execute("SELECT DISTINCT i FROM distances UNION SELECT DISTINCT j FROM distances")
                db_ids = sorted([int(r[0]) for r in cur.fetchall()])
            except Exception as err:
                logger.warning(err)
                db_ids = None
            conn.close()
            if db_ids is None or set(db_ids) != set(expected_ids):
                logger.warning(
                    "Distances DB %s incomplete; skipping distances-based task",
                    distances_db_path
                )
                return None
    except Exception as e:
        logger.debug("ID coverage check failed for %s: %s", distances_db_path, e)

    # First, try the shared helper which supports more DB formats
    try:
        from utils import load_distances
        distances_mat = load_distances(distances_db_path)
    except Exception as e:
        logger.debug("utils.load_distances failed: %s; falling back to direct sqlite read", e)
        distances_mat = None

    # Fallback: direct sqlite inspection for simple i,j,dist tables
    if distances_mat is None:
        try:
            conn = sqlite3.connect(distances_db_path)
            cur = conn.cursor()
            # Inspect table columns for 'component' or 'distance'/'dist'
            cur.execute("PRAGMA table_info(distances)")
            cols = [r[1] for r in cur.fetchall()]
            if not cols:
                raise RuntimeError("Table 'distances' not found or empty schema")

            # Prefer component/distance if present; otherwise 'dist'
            if 'component' in cols and 'distance' in cols:
                # Build dict of components
                df = None
                try:
                    import pandas as pd
                    df = pd.read_sql_query("SELECT i,j,component,distance FROM distances ORDER BY component, i, j", conn)
                    comps = sorted(df['component'].unique())
                    comp_mats = {}
                    unique_ids = np.unique(np.concatenate([df['i'].values, df['j'].values]))
                    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
                    n = len(unique_ids)
                    for comp in comps:
                        comp_df = df[df['component'] == comp]
                        mat = np.full((n, n), np.nan, dtype=float)
                        for _, row in comp_df.iterrows():
                            i = int(row['i']); j = int(row['j']); d = float(row['distance'])
                            mat[id_to_idx[i], id_to_idx[j]] = d
                        comp_mats[comp] = mat
                    distances_mat = comp_mats
                except Exception:
                    distances_mat = None
            else:
                dist_col = 'distance' if 'distance' in cols else ('dist' if 'dist' in cols else None)
                if dist_col is None:
                    distances_mat = None
                else:
                    # Read all rows into a DataFrame-like structure
                    import pandas as pd
                    df = pd.read_sql_query(f"SELECT i,j,{dist_col} as dist FROM distances ORDER BY i, j", conn)
                    rows = df.values
                    unique_ids = np.unique(np.concatenate([df['i'].values, df['j'].values]))
                    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
                    n = len(unique_ids)
                    mat = np.full((n, n), np.nan, dtype=float)
                    for i, j, d in rows:
                        mat[id_to_idx[int(i)], id_to_idx[int(j)]] = float(d)
                    distances_mat = mat
            conn.close()
        except Exception as e:
            logger.warning("Direct sqlite read failed for %s: %s", distances_db_path, e)
            return None

    # If we have per-component dict pick component based on target
    try:
        if isinstance(distances_mat, dict):
            comp_map = {
                'seg_loss': 'seg',
                'box_loss': 'box',
                'cls_loss': 'cls',
                'dfl_loss': 'dfl',
            }
            comp_name = comp_map.get(target)
            if comp_name and comp_name in distances_mat:
                D = distances_mat[comp_name]
                logger.debug("Selected component '%s' from distances DB", comp_name)
            elif 'total' in distances_mat:
                D = distances_mat['total']
                logger.debug("Selected 'total' component from distances DB")
            else:
                D = list(distances_mat.values())[0]
                logger.debug("Selected first available component from distances DB")
        else:
            D = distances_mat

        # Ensure square matrix before symmetrizing
        if D.shape[0] != D.shape[1]:
            raise ValueError(f"Distance matrix is not square: {D.shape}")

        # Symmetrize using nan-aware mean in case of missing entries
        D_sym = np.nanmean(np.stack([D, D.T]), axis=0)
        # ensure diagonal zero
        np.fill_diagonal(D_sym, 0.0)

        # Require a fully-observed matrix (no missing pairs). If any entry is
        # NaN after symmetrization, skip scheduling distances-based task.
        n_total = D_sym.size
        n_finite = np.isfinite(D_sym).sum()
        if n_finite != n_total:
            frac = float(n_finite) / float(n_total) if n_total > 0 else 0.0
            logger.warning(
                "Distances DB %s (target=%s) incomplete: %d/%d finite entries (%.3f%%); skipping distances-based task",
                distances_db_path,
                target,
                int(n_finite),
                int(n_total),
                frac * 100.0,
            )
            return None

        # All entries are finite -> return symmetric matrix
        logger.debug("Loaded symmetric distances from %s (target=%s)", distances_db_path, target)
        return D_sym.astype(np.float32)
    except Exception as e:
        logger.warning("Error processing distances matrix from %s: %s", distances_db_path, e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/grid_config.yaml')
    parser.add_argument('--db', default=None)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = args.db or db_utils.get_grid_db_path()
    db_utils.create_grid_db(db_path)

    embeddings = cfg.get('embeddings', [])
    targets = cfg.get('targets', [])
    preprocs = cfg.get('preproc', [])
    fss = cfg.get('feature_selection', [])
    extractors = cfg.get('extractors', [])

    knn_grid = build_knn_grid()

    emb_dir = "/CECI/home/ucl/irec/darimez/embeddings"
    dist_dir = "/CECI/home/ucl/irec/darimez/distances"

    tasks = []
    # load embeddings per model
    embeddings_cache = {}
    meta_cache = {}
    ids_cache = {}
    for emb in embeddings:
        if emb_dir is None:
            print(f"No embeddings_dir provided; skip loading {emb}")
            continue
        db_path_emb = os.path.join(emb_dir, f"{emb}.db")
        if not os.path.exists(db_path_emb):
            print(f"Embeddings DB not found: {db_path_emb}; skipping")
            continue
        X, meta, ids = db_utils.read_embeddings_db(db_path_emb)
        embeddings_cache[emb] = X
        meta_cache[emb] = meta
        ids_cache[emb] = ids

    # build folds once using first available embedding
    if not embeddings_cache:
        print("No embeddings loaded; aborting")
        return

    first_emb = next(iter(embeddings_cache))
    X0 = embeddings_cache[first_emb]
    n = X0.shape[0]
    seed = cfg.get('seed', 12345)
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(np.arange(n)):
        folds.append({'train_idx': train_idx, 'val_idx': val_idx})
    # persist folds
    db_utils.save_folds(db_path, seed, folds)

    # build tasks: embedding-based and precomputed-distance-based
    for emb in embeddings_cache.keys():
        X = embeddings_cache[emb]
        ids = ids_cache[emb]
        for target in targets:
            # Prefer extracting target values from a single reference embedding's metadata
            # (the first loaded); fall back to per-embedding meta if missing.
            ref_meta = meta_cache.get(first_emb, {})
            meta = meta_cache.get(emb, {})
            if target in ref_meta:
                y = ref_meta[target]
            elif target in meta:
                y = meta[target]
            else:
                print(f"Target {target} not found in embeddings meta for {emb} or reference {first_emb}; skipping")
                continue
            for preproc in preprocs:
                for fs in fss:
                    for ex in extractors:
                        # create a stable, task-unique run_id so we can persist bad runs
                        params_json = json.dumps(ex.get('params', {}), sort_keys=True)
                        params_hash = hashlib.md5(params_json.encode('utf-8')).hexdigest()[:8]
                        base_run = cfg.get('run_id', 'run0')
                        task_run_id = f"{base_run}_{emb}_{target}_{preproc}_{fs}_{ex['name']}_{params_hash}"

                        task = {
                            'run_id': task_run_id,
                            'embedding_model': emb,
                            'target': target,
                            'aggregation': 'image_mean',
                            'preproc': 'standard' if preproc == 'standard' else 'none',
                            'feature_selection': fs,
                            'extractor': ex['name'],
                            'extractor_params': ex.get('params', {}),
                            'knn_grid': knn_grid,
                            'use_precomputed_distances': False,
                        }

                        # Skip scheduling tasks that are recorded as bad (e.g., too few features / n_components issues)
                        try:
                            if bad_runs.is_bad_run(task_run_id):
                                print(f"Skipping task (known bad run): {task_run_id}")
                                continue
                        except Exception:
                            # if bad_runs lookup fails, proceed to schedule the task to avoid silent loss
                            pass

                        # Scheduler-level resume: skip tasks already present in the grid DB
                        try:
                            if db_utils.has_results(
                                db_path,
                                run_id=task_run_id,
                                embedding_model=emb,
                                target=target,
                                preproc='standard' if preproc == 'standard' else 'none',
                                feature_selection=fs,
                                extractor=ex['name'],
                                extractor_params=str(ex.get('params', {})),
                            ):
                                print(f"Skipping task (already in DB): {task_run_id}")
                                continue
                        except Exception:
                            # if DB lookup fails, schedule the task to avoid silent loss
                            pass

                        tasks.append((task, X, y, ids))

                        # also add precomputed-distance variant if distances present
                        if dist_dir is not None:
                            dist_path = os.path.join(dist_dir, f"{emb}.db")
                            if os.path.exists(dist_path):
                                D_sym = load_and_symmetrize_distances(dist_path, target=target, expected_ids=ids)
                                if D_sym is not None:
                                    task2 = dict(task)
                                    # For precomputed distances we run KNN directly (no preprocessing/extraction)
                                    task2['use_precomputed_distances'] = True
                                    task2['knn_grid'] = build_knn_grid_distances()  # Only n_neighbors for distances
                                    task2['feature_source'] = 'distances'
                                    # disable pipeline steps for distance-only runs
                                    task2['preproc'] = 'none'
                                    task2['feature_selection'] = None
                                    task2['extractor'] = None
                                    task2['extractor_params'] = {}
                                    # tag component name if available so downstream code can log/inspect
                                    # (note: the loader already selected the appropriate component when target provided)
                                    task2['distance_component'] = None
                                    tasks.append((task2, D_sym, y, ids))
                                else:
                                    print(f"Could not load/symmetrize distances from {dist_path}")
                            else:
                                print(f"Distances DB not found: {dist_path}")

    print(f"Prepared {len(tasks)} tasks.")

    # dispatch tasks with joblib; each task is (task_dict, X, y, ids)
    def _run_single(tup):
        task, X, y, ids = tup
        return worker_run(task, X, y, ids, folds, db_path, distances_db_path=(os.path.join(dist_dir, f"{task['embedding_model']}.db") if dist_dir else None))

    Parallel(n_jobs=args.n_jobs, backend='loky')(delayed(_run_single)(t) for t in tasks)


if __name__ == '__main__':
    main()
