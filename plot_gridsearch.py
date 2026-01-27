"""Plot GridSearch results from a gridsearch DB.

Usage:
    python vllmcluster/plot_gridsearch.py --db-path /path/to/gridsearch.db --metric spearman --out grid_plots.html --open

If --db-path is omitted, the script will try (in order):
  - `db_utils.get_grid_db_path()` (if available)
  - $CECIHOME/gridsearch.db
  - ~/gridsearch.db
  - /CECI/home/ucl/irec/darimez/gridsearch.db

Output:
  Writes an interactive HTML file containing one or two plots (box and scatter).
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from typing import Optional

import pandas as pd
import plotly.express as px

try:
    import db_utils
    _HAS_DB_UTILS = True
except Exception:
    _HAS_DB_UTILS = False


def resolve_db_path(db_path: Optional[str]) -> str:
    if db_path:
        return db_path

    if _HAS_DB_UTILS:
        try:
            default = db_utils.get_grid_db_path()
            if default and os.path.exists(default):
                return default
        except Exception:
            pass

    ceci = os.environ.get("CECIHOME")
    if ceci:
        p = os.path.join(ceci, "grid_search.db")
        if os.path.exists(p):
            return p

    local = os.path.expanduser("~/grid_search.db")
    if os.path.exists(local):
        return local

    # last-resort fallback used in this project
    return "/CECI/home/ucl/irec/darimez/grid_search.db"


def load_grid_df(db_path: str) -> pd.DataFrame:
    if _HAS_DB_UTILS:
        try:
            return db_utils.load_grid_results(db_path)
        except Exception:
            pass

    # Fallback: try to read a common table name
    import sqlite3
    conn = sqlite3.connect(db_path)
    # Try a couple of likely table names used in this project
    for tbl in ("grid_results", "gridsearch", "grid_search", "results"):
        try:
            df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
            conn.close()
            return df
        except Exception:
            continue
    conn.close()
    raise RuntimeError(f"Could not read grid results from {db_path}; no known table found")


def pick_metric(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    # common heuristic
    for c in ("spearman", "corr", "correlation/ARI", "mae", "mae/accuracy", "r2"):
        if c in df.columns:
            return c
    # else pick a numeric column
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        return numeric[0]
    raise RuntimeError("No suitable numeric metric column found in grid DB")


def make_plots(df: pd.DataFrame, metric: str):
    # normalize typical column names used in this repo
    if "knn_n" in df.columns and "k" not in df.columns:
        df = df.copy()
        df["k"] = df["knn_n"]
    if "embedding_model" in df.columns and "model_full" not in df.columns:
        df = df.copy()
        df["model_full"] = df["embedding_model"].astype(str)

    # Box plot per k with colour per model
    if "k" in df.columns and "model_full" in df.columns:
        fig_box = px.box(df, x="k", y=metric, color="model_full", points="outliers", title=f"{metric} by k and model")
    else:
        # fallback: box per model
        fig_box = px.box(df, x="model_full" if "model_full" in df.columns else None, y=metric, points="outliers", title=f"{metric} by model")

    # Scatter of metric vs k
    if "k" in df.columns:
        fig_scatter = px.scatter(df, x="k", y=metric, color="model_full" if "model_full" in df.columns else None, hover_data=[col for col in ("model_full", "target") if col in df.columns], title=f"{metric} vs k")
    else:
        fig_scatter = None

    return fig_box, fig_scatter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", type=str, default=None, help="Path to gridsearch DB")
    p.add_argument("--metric", type=str, default=None, help="Metric to plot (auto-detected when omitted)")
    p.add_argument("--out", type=str, default=None, help="Output HTML filename (default: grid_plots_<metric>.html)")
    p.add_argument("--open", action="store_true", help="Open the resulting HTML in a browser after writing")
    args = p.parse_args()

    db_path = resolve_db_path(args.db_path)
    if not os.path.exists(db_path):
        raise SystemExit(f"Gridsearch DB not found at {db_path}")

    df = load_grid_df(db_path)
    if df.empty:
        raise SystemExit("Gridsearch DB loaded but contains no rows")

    metric = pick_metric(df, args.metric)
    print(f"Using metric: {metric}")

    fig_box, fig_scatter = make_plots(df, metric)

    out = args.out or f"grid_plots_{metric}.html"

    # Create a simple HTML that embeds both plots
    from plotly.io import to_html
    parts = [to_html(fig_box, include_plotlyjs="cdn", full_html=False)]
    if fig_scatter is not None:
        parts.append(to_html(fig_scatter, include_plotlyjs=False, full_html=False))

    html = "\n<hr>\n".join(parts)
    html = f"<!doctype html><html><head><meta charset=\"utf-8\"><title>GridSearch plots</title></head><body>{html}</body></html>"

    with open(out, "w") as fh:
        fh.write(html)

    print(f"Wrote plots to {out}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
