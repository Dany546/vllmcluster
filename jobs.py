"""Job worker wrappers for gridsearch.

`worker_run` is implemented to accept preloaded embeddings and run a single
pipeline configuration across all folds. It writes per-fold results to the
grid_search DB using `db_utils.insert_grid_result`.
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .pipelines import FeaturePipeline, NoFeaturesLeft
from . import db_utils


def worker_run(
    config: Dict[str, Any],
    X: Optional[np.ndarray],
    y: Optional[np.ndarray],
    ids: Optional[List[int]],
    folds: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    distances_db_path: Optional[str] = None,
):
    """Run one pipeline configuration across provided folds.

    `config` contains keys: preproc, feature_selection (str or None), extractor (name), extractor_params (dict), knn_grid (list of dicts), embedding_model, target.
    """
    db_path = db_path or db_utils.get_grid_db_path()
    results = []
    for fi, f in enumerate(folds):
        train_idx = f['train_idx']
        val_idx = f['val_idx']
        # prepare train/val sets either from X or from precomputed distances
        if config.get('use_precomputed_distances'):
            # ids must be provided to map to distances DB
            if ids is None or distances_db_path is None:
                db_utils.insert_skipped(db_path, {
                    'run_id': config.get('run_id'),
                    'embedding_model': config.get('embedding_model'),
                    'target': config.get('target'),
                    'aggregation': config.get('aggregation'),
                    'preproc': config.get('preproc'),
                    'feature_selection': config.get('feature_selection'),
                    'extractor': config.get('extractor'),
                    'extractor_params': str(config.get('extractor_params', {})),
                    'fold': fi,
                    'knn_n': None,
                    'knn_metric': None,
                    'reason': 'no_ids_or_dist_db',
                })
                continue
            train_ids = [ids[i] for i in train_idx]
            val_ids = [ids[i] for i in val_idx]
        else:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

        # build pipeline steps (only used for embedding-based pipelines)
        steps = []
        if config.get('preproc') == 'standard':
            from .pipelines import StandardScalerWrapper

            steps.append(StandardScalerWrapper())

        fs = config.get('feature_selection')
        if fs and fs.startswith('spearman-'):
            thr = float(fs.split('-')[-1])
            from .pipelines import SpearmanFeatureSelector

            steps.append(SpearmanFeatureSelector(thr))

        # extractor
        extractor = config.get('extractor')
        extractor_params = config.get('extractor_params', {})
        if extractor == 'PLS':
            from .pipelines import PLSExtractor

            steps.append(PLSExtractor(**extractor_params))
        elif extractor == 'KPCA':
            from .pipelines import KernelPCAExtractor

            steps.append(KernelPCAExtractor(**extractor_params))
        elif extractor == 'TSNE':
            from .pipelines import TSNEExtractor

            steps.append(TSNEExtractor(**extractor_params))
        elif extractor == 'UMAP':
            from .pipelines import UMAPExtractor

            steps.append(UMAPExtractor(**extractor_params))
        else:
            raise RuntimeError(f"Unknown extractor: {extractor}")

        # for distance-based pipelines we don't transform embeddings; we will read distances later
        if config.get('use_precomputed_distances'):
            # prepare distances matrices
            reader = db_utils.DistancesDBReader(distances_db_path)
            D_train = reader.get_pairwise_distance_matrix(train_ids)
            D_val_train = reader.get_cross_distance_matrix(val_ids, train_ids)

        else:
            pipeline = FeaturePipeline(steps)

            # fit_transform on train only
            try:
                X_train_t = pipeline.fit_transform(X_train, y_train)
            except NoFeaturesLeft:
                db_utils.insert_skipped(db_path, {
                    'run_id': config.get('run_id'),
                    'embedding_model': config.get('embedding_model'),
                    'target': config.get('target'),
                    'aggregation': config.get('aggregation'),
                    'preproc': config.get('preproc'),
                    'feature_selection': config.get('feature_selection'),
                    'extractor': extractor,
                    'extractor_params': str(extractor_params),
                    'fold': fi,
                    'knn_n': None,
                    'knn_metric': None,
                    'reason': 'no_features',
                })
                continue

            # transform val
            X_val_t = pipeline.transform(X_val)

        # run KNN grid
        for knn_cfg in config.get('knn_grid', []):
            n_neighbors = knn_cfg['n_neighbors']
            metric = knn_cfg['metric']
            # safety
            if config.get('use_precomputed_distances'):
                if n_neighbors > D_train.shape[0]:
                    db_utils.insert_skipped(db_path, {
                        'run_id': config.get('run_id'),
                        'embedding_model': config.get('embedding_model'),
                        'target': config.get('target'),
                        'aggregation': config.get('aggregation'),
                        'preproc': config.get('preproc'),
                        'feature_selection': config.get('feature_selection'),
                        'extractor': extractor,
                        'extractor_params': str(extractor_params),
                        'fold': fi,
                        'knn_n': n_neighbors,
                        'knn_metric': metric,
                        'reason': 'n_neighbors_gt_ntrain',
                    })
                    continue
                knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='precomputed', weights='distance')
                knn.fit(D_train, y_train)
                y_pred = knn.predict(D_val_train)
            else:
                if n_neighbors > X_train_t.shape[0]:
                    db_utils.insert_skipped(db_path, {
                        'run_id': config.get('run_id'),
                        'embedding_model': config.get('embedding_model'),
                        'target': config.get('target'),
                        'aggregation': config.get('aggregation'),
                        'preproc': config.get('preproc'),
                        'feature_selection': config.get('feature_selection'),
                        'extractor': extractor,
                        'extractor_params': str(extractor_params),
                        'fold': fi,
                        'knn_n': n_neighbors,
                        'knn_metric': metric,
                        'reason': 'n_neighbors_gt_ntrain',
                    })
                    continue
                knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric, weights='distance')
                knn.fit(X_train_t, y_train)
                y_pred = knn.predict(X_val_t)

            # compute metrics
            from scipy.stats import spearmanr
            from sklearn.metrics import r2_score, mean_absolute_error

            try:
                spearman = float(spearmanr(y_val, y_pred).correlation)
            except Exception:
                spearman = float('nan')

            r2 = float(r2_score(y_val, y_pred))
            mae = float(mean_absolute_error(y_val, y_pred))

            row = {
                'run_id': config.get('run_id'),
                'embedding_model': config.get('embedding_model'),
                'target': config.get('target'),
                'aggregation': config.get('aggregation'),
                'preproc': config.get('preproc'),
                'feature_selection': config.get('feature_selection'),
                'extractor': extractor,
                'extractor_params': str(extractor_params),
                'knn_n': n_neighbors,
                'knn_metric': metric,
                'fold': fi,
                'spearman': spearman,
                'r2': r2,
                'mae': mae,
                'status': 'ok',
            }
            db_utils.insert_grid_result(db_path, row)

    return results
