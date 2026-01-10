"""Reporting utilities for gridsearch results.

Provides aggregation and CSV export of per-fold metrics stored in
`grid_search.db`.
"""
from __future__ import annotations

import os
import sqlite3
import csv
from typing import Optional, List, Dict, Any
import statistics


def aggregate_results(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    db_path = db_path or os.path.join(os.getcwd(), 'grid_search.db')
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT embedding_model, target, preproc, feature_selection, extractor, extractor_params, knn_n, knn_metric, spearman, r2, mae FROM grid_results WHERE status='ok'")
    rows = cur.fetchall()
    con.close()

    # group by keys
    groups: Dict[tuple, List[tuple]] = {}
    for r in rows:
        key = tuple(r[:8])
        vals = tuple(r[8:])
        groups.setdefault(key, []).append(vals)

    out = []
    for key, vals in groups.items():
        spears = [v[0] for v in vals if v[0] is not None]
        r2s = [v[1] for v in vals if v[1] is not None]
        maes = [v[2] for v in vals if v[2] is not None]
        row = {
            'embedding_model': key[0],
            'target': key[1],
            'preproc': key[2],
            'feature_selection': key[3],
            'extractor': key[4],
            'extractor_params': key[5],
            'knn_n': key[6],
            'knn_metric': key[7],
            'n_folds': len(vals),
            'spearman_mean': statistics.mean(spears) if spears else None,
            'spearman_std': statistics.pstdev(spears) if spears else None,
            'r2_mean': statistics.mean(r2s) if r2s else None,
            'r2_std': statistics.pstdev(r2s) if r2s else None,
            'mae_mean': statistics.mean(maes) if maes else None,
            'mae_std': statistics.pstdev(maes) if maes else None,
        }
        out.append(row)
    return out


def export_csv(aggregated: List[Dict[str, Any]], out_csv: Optional[str] = None) -> str:
    out_csv = out_csv or os.path.join(os.getcwd(), 'outputs', 'grid_search_summary.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ['embedding_model', 'target', 'preproc', 'feature_selection', 'extractor', 'extractor_params', 'knn_n', 'knn_metric', 'n_folds', 'spearman_mean', 'spearman_std', 'r2_mean', 'r2_std', 'mae_mean', 'mae_std']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in aggregated:
            # convert None to empty
            clean = {k: ('' if row.get(k) is None else row.get(k)) for k in keys}
            w.writerow(clean)
    return out_csv


def top_configs(db_path: Optional[str] = None, top_n: int = 20) -> List[Dict[str, Any]]:
    ag = aggregate_results(db_path)
    # sort by spearman_mean desc
    ag_sorted = sorted([a for a in ag if a['spearman_mean'] is not None], key=lambda x: x['spearman_mean'], reverse=True)
    return ag_sorted[:top_n]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    agg = aggregate_results(args.db)
    path = export_csv(agg, args.out)
    print('Exported summary to', path)
