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

from . import db_utils
from .jobs import worker_run


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='vllmcluster/configs/grid_config.yaml')
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

    emb_dir = args.__dict__.get('embeddings_dir') or cfg.get('embeddings_dir')
    dist_dir = args.__dict__.get('distances_dir') or cfg.get('distances_dir')

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
            # extract target values from meta if present
            meta = meta_cache[emb]
            if target not in meta:
                print(f"Target {target} not found in embeddings meta for {emb}; skipping")
                continue
            y = meta[target]
            for preproc in preprocs:
                for fs in fss:
                    for ex in extractors:
                        task = {
                            'run_id': cfg.get('run_id', 'run0'),
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
                        tasks.append((task, X, y, ids))

                        # also add precomputed-distance variant if distances present
                        if dist_dir is not None:
                            dist_path = os.path.join(dist_dir, f"{emb}_distances.db")
                            if os.path.exists(dist_path):
                                task2 = dict(task)
                                task2['use_precomputed_distances'] = True
                                tasks.append((task2, None, y, ids))

    print(f"Prepared {len(tasks)} tasks.")

    # dispatch tasks with joblib; each task is (task_dict, X, y, ids)
    def _run_single(tup):
        task, X, y, ids = tup
        return worker_run(task, X, y, ids, folds, db_path, distances_db_path=(os.path.join(dist_dir, f"{task['embedding_model']}_distances.db") if dist_dir else None))

    Parallel(n_jobs=args.n_jobs, backend='loky')(delayed(_run_single)(t) for t in tasks)


if __name__ == '__main__':
    main()
