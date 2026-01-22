#!/usr/bin/env python3
"""
Check embeddings and distances DBs for:
 - expected number of entries
 - metric columns not all zero
 - loss columns not excessively high
 - distances components not all zero
"""

import argparse
import sqlite3
import sys, os
import struct
import numpy as np

DEFAULT_METRICS = ["hit_freq", "mean_iou", "mean_conf", "mean_dice"]
DEFAULT_LOSSES = ["box_loss", "cls_loss", "dfl_loss", "seg_loss"]
DEFAULT_THRESHOLDS = {"box_loss": 1.0, "cls_loss": 1.0, "dfl_loss": 1.0, "seg_loss": 1.0}

EPS = 1e-8

def parse_numeric(v):
    """Try to convert a DB cell to float. Returns float or None for non-numeric."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, bytes):
        # Try little-endian float32 or float64 if size matches
        if len(v) == 4:
            try:
                return float(struct.unpack("<f", v)[0])
            except Exception:
                pass
        if len(v) == 8:
            try:
                return float(struct.unpack("<d", v)[0])
            except Exception:
                pass
        # Try interpreting as ascii float string
        try:
            s = v.decode('utf-8')
            return float(s)
        except Exception:
            pass
        # Try numpy buffer -> mean (arrays stored as bytes)
        try:
            a = np.frombuffer(v, dtype=np.float32)
            if a.size > 0:
                # Return single value if scalar-like else mean
                return float(a.ravel()[0]) if a.size == 1 else float(a.mean())
        except Exception:
            pass
    return None

def get_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def load_column_floats(conn, table, column):
    cur = conn.execute(f"SELECT {column} FROM {table}")
    vals = []
    for (v,) in cur.fetchall():
        x = parse_numeric(v)
        vals.append(np.nan if x is None else x)
    return np.array(vals, dtype=float)

def check_embeddings(conn, expected_count, metric_cols, loss_cols, loss_thresholds, loss_fraction_allowed):
    report = {"ok": True, "messages": []}
    # Count rows and distinct img_id

    cur = conn.execute("SELECT COUNT(*) FROM embeddings")
    total_rows = cur.fetchone()[0]
    report["total_rows"] = total_rows
    if total_rows < expected_count:
        report["ok"] = False
        report["messages"].append(f"Embeddings table has {total_rows} rows (< expected {expected_count})")

    # Distinct img_id
    try:
        cur = conn.execute("SELECT COUNT(DISTINCT img_id) FROM embeddings")
        distinct_img = cur.fetchone()[0]
        if distinct_img < expected_count:
            report["ok"] = False
            report["messages"].append(f"Embeddings table has {distinct_img} distinct img_id (< expected {expected_count})")
    except Exception:
        report["messages"].append("Could not compute DISTINCT img_id (column may be missing)")

    cols = get_columns(conn, "embeddings")

    for c in metric_cols:
        if c not in cols:
            report["messages"].append(f"Metric column `{c}` not found in embeddings")
            continue
        arr = load_column_floats(conn, "embeddings", c)
        n_total = arr.size
        n_zero = np.sum(np.isnan(arr) | (np.abs(arr) <= EPS))
        if n_zero == n_total:
            report["ok"] = False
            report["messages"].append(f"Metric column `{c}` is all zeros or NULL ({n_zero}/{n_total})")

    for c in loss_cols:
        if c not in cols:
            report["messages"].append(f"Loss column `{c}` not found in embeddings (OK if model type doesn't include it)")
            continue
        arr = load_column_floats(conn, "embeddings", c)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            report["messages"].append(f"Loss column `{c}` has only NULLs")
            continue
        med = float(np.median(arr))
        p95 = float(np.nanpercentile(arr, 95))
        n_high = np.sum(arr > loss_thresholds.get(c, 1.0))
        frac_high = n_high / max(1, arr.size)
        if med > loss_thresholds.get(c, 1.0):
            report["ok"] = False
            report["messages"].append(f"Loss `{c}` median {med:.3f} > threshold {loss_thresholds.get(c):.3f}")
        if frac_high > loss_fraction_allowed:
            report["ok"] = False
            report["messages"].append(f"Loss `{c}` high values >{loss_thresholds.get(c):.3f} in {frac_high*100:.1f}% (> {loss_fraction_allowed*100:.1f}%) rows (p95={p95:.3f})")

    return report

def check_distances(conn):
    report = {"ok": True, "messages": []}
    # Check table exists
    try:
        cur = conn.execute("SELECT COUNT(*) FROM distances")
        total = cur.fetchone()[0]
    except Exception as e:
        return {"ok": False, "messages": [f"Distances table not present or inaccessible: {e}"]}

    cur = conn.execute("SELECT DISTINCT component FROM distances")
    components = [r[0] for r in cur.fetchall()]
    if not components:
        return {"ok": False, "messages": ["No components found in distances table"]}

    for comp in components:
        cur = conn.execute("SELECT distance FROM distances WHERE component = ?", (comp,))
        arr = np.array([parse_numeric(r[0]) for r in cur.fetchall()], dtype=float)
        if arr.size == 0:
            report["ok"] = False
            report["messages"].append(f"Component `{comp}` has no rows")
            continue
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            report["ok"] = False
            report["messages"].append(f"Component `{comp}` distance values are all NULL")
            continue
        if np.all(np.abs(arr) <= EPS):
            report["ok"] = False
            report["messages"].append(f"Component `{comp}` distances are all zero")
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-db", required=True)
    p.add_argument("--distances-db", required=True)
    p.add_argument("--expected-count", type=int, default=5000)
    p.add_argument("--loss-threshold", type=float, default=None,
                   help="global loss threshold (overrides per-loss defaults if set)")
    p.add_argument("--loss-fraction-allowed", type=float, default=0.05,
                   help="fraction of rows allowed to exceed loss threshold before flagging")
    args = p.parse_args()

    loss_thresholds = DEFAULT_THRESHOLDS.copy()
    if args.loss_threshold is not None:
        for k in loss_thresholds:
            loss_thresholds[k] = args.loss_threshold

    if not os.path.exists(args.embeddings_db):
        print(f"Embeddings DB not found: {args.embeddings_db}")
        return 1
    
    emb_conn = sqlite3.connect(args.embeddings_db)
    emb_report = check_embeddings(emb_conn, args.expected_count, DEFAULT_METRICS, DEFAULT_LOSSES, loss_thresholds, args.loss_fraction_allowed)
    emb_conn.close()

    if not os.path.exists(args.distances_db):
        print(f"Distances DB not found: {args.distances_db}")
        return 1
    
    dist_conn = sqlite3.connect(args.distances_db)
    dist_report = check_distances(dist_conn)
    dist_conn.close()

    ok = emb_report["ok"] and dist_report["ok"]

    print("=== EMBEDDINGS CHECK ===")
    for k,v in emb_report.items():
        if k == "messages":
            for m in v:
                print(" -", m)
        elif k != "ok":
            print(f" {k}: {v}")

    print("\n=== DISTANCES CHECK ===")
    for m in dist_report.get("messages", []):
        print(" -", m)

    if ok:
        print("\n✅ All checks passed")
        return 0
    else:
        print("\n❌ Some checks failed. See messages above.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
