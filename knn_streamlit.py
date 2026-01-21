"""
Per-cell Streamlit viewer for KNN results.

Features added in this patch:
- Grid of cells (rows x cols)
- Per-cell configuration (model, target, metric, k values, title, legend)
- HTML / PNG export per cell
- Cached DB load for speed

Run:
    streamlit run knn_streamlit.py -- --db-path /path/to/knn_results.db
    streamlit run knn_streamlit.py -- --local
"""

import argparse
import sqlite3
import os
import io
import base64
from typing import List, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import db_utils


def get_db_path() -> str:
    """Return a sensible default KNN DB path for Streamlit use.

    Prefer a local copy at ~/knn_results.db if present, otherwise fall
    back to the CECI home location used on the cluster.
    """
    # Prefer an explicit CECI home path when available
    ceci = os.environ.get("CECIHOME")
    if ceci:
        ceci_path = os.path.join(ceci, "knn_results.db")
        if os.path.exists(ceci_path):
            return ceci_path

    # Prefer a local copy in the user's home directory
    local_path = os.path.expanduser("~/knn_results.db")
    if os.path.exists(local_path):
        return local_path

    # If CECIHOME is set but file doesn't exist, return the expected CECI path
    if ceci:
        return os.path.join(ceci, "knn_results.db")

    # Fallback to home path (may be overridden by sidebar)
    return local_path



@st.cache_data
def load_knn_results(db_path: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM knn_results", conn)
        conn.close()
        if "distance_metric" in df.columns:
            df["model_full"] = df["model"].astype(str) + "_" + df["distance_metric"].astype(str)
        else:
            df["model_full"] = df["model"].astype(str)
        return df
    except Exception as e:
        st.error(f"Failed to load DB: {e}")
        return pd.DataFrame()


@st.cache_data
def load_grid_results(db_path: str) -> pd.DataFrame:
    try:
        return db_utils.load_grid_results(db_path)
    except Exception as e:
        st.error(f"Failed to load grid DB: {e}")
        return pd.DataFrame()


def make_cell_figure(df: pd.DataFrame,
                     models: List[str],
                     target: str,
                     metric: str,
                     k_values: Optional[List[int]],
                     title: Optional[str],
                     show_legend: bool,
                     y_range: Optional[List[float]]):
    fig = go.Figure()
    if df.empty or not models:
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    # If the requested metric is not in the dataframe, show a clear message
    if metric not in df.columns:
        avail = ", ".join(sorted(df.columns.tolist()))
        fig.add_annotation(
            text=f"Metric '{metric}' not found in data. Available columns: {avail}",
            showarrow=False,
            xref='paper', yref='paper', x=0.5, y=0.5
        )
        return fig

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52BE80"
    ]

    # ensure k_values is list
    if k_values:
        df = df[df["k"].isin(k_values)]

    for i, model in enumerate(models):
        sub = df[(df["model_full"] == model) & (df["target"] == target)]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                x=sub["k"],
                y=sub[metric],
                name=model,
                marker_color=colors[i % len(colors)],
                boxmean=True,
                visible=True,
            )
        )

    fig.update_layout(
        title=title or f"{metric.upper()} - {target}",
        xaxis_title="k neighbors",
        yaxis_title=metric,
        template="plotly_white",
        boxmode="group",
        height=420,
        showlegend=show_legend,
    )

    if y_range and len(y_range) == 2:
        fig.update_yaxes(range=y_range)

    return fig


def fig_to_downloads(fig: go.Figure):
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    try:
        png = fig.to_image(format="png")
    except Exception:
        png = None
    return html, png


st.set_page_config(page_title="KNN Results Explorer", layout="wide")
st.title("📊 KNN Results Explorer — Per-cell Grid")

# Load DB defaults
# determine a sensible default DB path
db_path = get_db_path()

# Allow user to override KNN DB path from the sidebar
knn_db_path = st.sidebar.text_input("KNN DB path", value=db_path)
if st.sidebar.button("Reload KNN DB"):
    st.experimental_rerun()

# load KNN results early so `df` is defined for downstream checks
resolved_knn = os.path.expanduser(knn_db_path) if knn_db_path else ""
if resolved_knn and os.path.exists(resolved_knn):
    df = load_knn_results(resolved_knn)
else:
    st.warning(f"KNN DB not found at {resolved_knn!s}; KNN pages may be limited. Use GridSearch page to view grid_search.db results.")
    df = pd.DataFrame(columns=["k", "model", "target", "model_full", "corr", "mae", "r2"])

# Grid-search DB selector (optional)
grid_db_default = db_utils.get_grid_db_path()
grid_db_path = st.sidebar.text_input("Grid search DB path", value=grid_db_default)
if st.sidebar.button("Reload grid DB"):
    st.experimental_rerun()

# If KNN DB failed to load, don't stop the app; allow GridSearch page to work
if df.empty:
    st.warning("No KNN results DB found; KNN pages may be limited. Use GridSearch page to view grid_search.db results.")
    df = pd.DataFrame(columns=["k", "model", "target", "model_full", "corr", "mae", "r2"])


st.sidebar.header("Layout & Global Filters")
n_rows = st.sidebar.number_input("Grid rows", min_value=1, max_value=4, value=1, key="grid_rows")
n_cols = st.sidebar.number_input("Grid cols", min_value=1, max_value=4, value=2, key="grid_cols")

# Page selector: Plots, Statistics or GridSearch
page = st.sidebar.radio("Page", options=["Plots", "Statistics", "GridSearch"], index=0)

# Data source selector for plotting: choose which DB to use for the Plots page
data_source = st.sidebar.selectbox("Data source", options=["KNN results", "Grid search"], index=0)

# Load/normalize plotting DataFrame depending on data source
if data_source == "KNN results":
    resolved_knn = os.path.expanduser(knn_db_path) if knn_db_path else ""
    if resolved_knn and os.path.exists(resolved_knn):
        df_plot = load_knn_results(resolved_knn)
    else:
        st.warning(f"KNN DB not found at {resolved_knn!s}; using empty dataset.")
        df_plot = pd.DataFrame()
    # ensure consistent column names expected by plotting helpers
    if not df_plot.empty and "model_full" not in df_plot.columns:
        if "distance_metric" in df_plot.columns:
            df_plot["model_full"] = df_plot["model"].astype(str) + "_" + df_plot["distance_metric"].astype(str)
        else:
            df_plot["model_full"] = df_plot["model"].astype(str)
    # normalize common metric column names to expected keys
    if not df_plot.empty:
        if "correlation/ARI" in df_plot.columns and "corr" not in df_plot.columns:
            df_plot["corr"] = df_plot["correlation/ARI"]
        if "mae/accuracy" in df_plot.columns and "mae" not in df_plot.columns:
            df_plot["mae"] = df_plot["mae/accuracy"]
else:
    resolved_grid = os.path.expanduser(grid_db_path) if grid_db_path else ""
    if resolved_grid and os.path.exists(resolved_grid):
        df_grid = load_grid_results(resolved_grid)
    else:
        st.warning(f"Grid search DB not found at {resolved_grid!s}; GridSearch may be limited.")
        df_grid = pd.DataFrame()
    df_plot = df_grid.copy()
    # normalize column names to match knn_results expectations
    if "knn_n" in df_plot.columns:
        df_plot["k"] = df_plot["knn_n"]
    if "embedding_model" in df_plot.columns:
        df_plot["model_full"] = df_plot["embedding_model"].astype(str)
    # map spearman to corr so existing UI labels work
    if "spearman" in df_plot.columns and "corr" not in df_plot.columns:
        df_plot["corr"] = df_plot["spearman"]

# Global k options (from selected data source)
k_options = sorted(df_plot["k"].unique().tolist()) if (not df_plot.empty and "k" in df_plot.columns) else []
global_k = st.sidebar.multiselect("Global k values", options=k_options, default=k_options)

# Available models (from selected data source)
models_all = sorted(df_plot["model_full"].unique().tolist()) if (not df_plot.empty and "model_full" in df_plot.columns) else []
global_models = st.sidebar.multiselect("Global models", options=models_all, default=models_all)

st.sidebar.markdown("---")
st.sidebar.write("Per-cell fallbacks (cells override when non-empty)")
# available metrics for the selected data source
available_metrics = [c for c in ["corr", "spearman", "mae", "r2"] if c in df_plot.columns]
if not available_metrics:
    # fallback to any numeric columns
    available_metrics = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])] if not df_plot.empty else []
global_metric = st.sidebar.selectbox("Global metric", options=available_metrics or [""], index=0)
targets_list = sorted(df_plot["target"].unique().tolist()) if (not df_plot.empty and "target" in df_plot.columns) else []
if not targets_list:
    targets_list = ["(no targets)"]
global_target = st.sidebar.selectbox("Global target", options=targets_list, index=0)
global_query = st.sidebar.text_input("Global trace filter (substring)", value="", help="Case-insensitive substring to filter traces")

# Presets (simple in-session)
if "knn_presets" not in st.session_state:
    st.session_state["knn_presets"] = {}

with st.sidebar.expander("Save / Load Preset"):
    pname = st.text_input("Preset name")
    if st.button("Save preset") and pname:
        st.session_state["knn_presets"][pname] = {"rows": n_rows, "cols": n_cols}
        st.success("Saved")
    load_p = st.selectbox("Load preset", options=[""] + list(st.session_state["knn_presets"].keys()))
    if load_p:
        val = st.session_state["knn_presets"][load_p]
        n_rows = val.get("rows", n_rows)
        n_cols = val.get("cols", n_cols)
        st.experimental_rerun()

# Grid rendering
if page == "Statistics":
    st.header("📈 Statistics")
    # Apply global filters to produce a summary similar to the previous implementation
    df_summary = df_plot.copy()
    if global_k:
        df_summary = df_summary[df_summary["k"].isin(global_k)]
    if global_models:
        df_summary = df_summary[df_summary["model_full"].isin(global_models)]
    if global_query and "model_full" in df_summary.columns:
        q = global_query.lower()
        df_summary = df_summary[df_summary["model_full"].str.lower().str.contains(q)]

    # Per-column regex filters (allow combining multiple column filters)
    with st.expander("Filters (apply regex per column)", expanded=False):
        cols = df_summary.columns.tolist()
        sel_cols = st.multiselect("Columns to filter", options=cols, key="stats_filter_cols")
        filter_vals = {}
        for col in sel_cols:
            v = st.text_input(f"Regex for '{col}'", value="", key=f"stats_filter_{col}")
            if v:
                filter_vals[col] = v

        if st.button("Clear filters", key="stats_clear_filters"):
            for col in sel_cols:
                st.session_state.pop(f"stats_filter_{col}", None)
            st.session_state["stats_filter_cols"] = []
            st.experimental_rerun()

    # Apply user-provided column regex filters (case-insensitive)
    for col, regex in (filter_vals.items() if isinstance(filter_vals, dict) else []):
        try:
            df_summary = df_summary[df_summary[col].astype(str).str.contains(regex, case=False, na=False, regex=True)]
        except Exception as e:
            st.warning(f"Invalid regex for column {col}: {e}")

    # Show filtered raw rows and aggregated summary
    st.write("### Filtered Rows")
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    st.write("### Aggregated Data Summary")
    ag_cols = {m: ["mean", "std"] for m in available_metrics}
    if ag_cols:
        st.dataframe(
            df_summary.groupby(["model_full", "target", "k"]).agg(ag_cols).round(4),
            use_container_width=True,
        )
    else:
        st.write("No numeric metrics available to summarize.")

elif page == "GridSearch":
    st.header("🔎 Grid Search Results")
    resolved_grid = os.path.expanduser(grid_db_path) if grid_db_path else ""
    if resolved_grid and os.path.exists(resolved_grid):
        df_grid = load_grid_results(resolved_grid)
    else:
        st.info("No grid search results found or DB path invalid")
        df_grid = pd.DataFrame()
    if df_grid.empty:
        st.info("No grid search results found or DB path invalid")
    else:
        st.write("### Raw grid results")
        st.dataframe(df_grid, use_container_width=True)

        st.write("### Mean spearman by embedding model / target")
        if 'spearman' in df_grid.columns:
            agg = df_grid.groupby(['embedding_model', 'target'], dropna=False)['spearman'].mean().reset_index()
            agg = agg.sort_values('spearman', ascending=False)
            st.dataframe(agg, use_container_width=True)
            try:
                chart_df = agg.groupby('embedding_model')['spearman'].mean()
                st.bar_chart(chart_df)
            except Exception:
                pass
        else:
            st.write("No 'spearman' column available to aggregate")

else:
    st.write(f"Showing plots: {n_rows} x {n_cols} (source: {data_source})")
    cells = n_rows * n_cols
    cell_idx = 0
    for r in range(n_rows):
        row_cols = st.columns(n_cols)
        for c in range(n_cols):
            with row_cols[c]:
                with st.expander(f"Cell {cell_idx + 1} settings", expanded=False):
                    chosen_models = st.multiselect(
                        f"Models (cell {cell_idx+1})",
                        options=models_all,
                        default=[],
                        key=f"models_{cell_idx}"
                    )

                    chosen_target = st.selectbox(
                        f"Target (cell {cell_idx+1})",
                        options=["(use global)"] + sorted(df_plot["target"].unique().tolist()) if (not df_plot.empty and "target" in df_plot.columns) else ["(use global)"],
                        index=0,
                        key=f"target_{cell_idx}"
                    )

                    chosen_metric = st.selectbox(
                        f"Metric (cell {cell_idx+1})",
                        options=["(use global)"] + available_metrics,
                        index=0,
                        key=f"metric_{cell_idx}"
                    )

                    k_sel = st.multiselect(
                        f"k values (cell {cell_idx+1})",
                        options=k_options,
                        default=[],
                        key=f"k_{cell_idx}"
                    )

                    title = st.text_input(f"Title (cell {cell_idx+1})", value="", key=f"title_{cell_idx}")
                    show_legend = st.checkbox(f"Show legend (cell {cell_idx+1})", value=False, key=f"legend_{cell_idx}")
                    y_min = st.number_input(f"Y min (cell {cell_idx+1})", value=float('nan'), key=f"ymin_{cell_idx}")
                    y_max = st.number_input(f"Y max (cell {cell_idx+1})", value=float('nan'), key=f"ymax_{cell_idx}")

                # derive effective values (per-cell overrides fall back to global)
                effective_models = chosen_models if chosen_models else global_models
                effective_target = chosen_target if chosen_target != "(use global)" else global_target
                effective_metric = chosen_metric if chosen_metric != "(use global)" else global_metric
                effective_k = k_sel if k_sel else global_k
                effective_title = title or f"{effective_metric.upper()} - {effective_target}"

                # apply global query as filter for model names
                if global_query:
                    q = global_query.lower()
                    effective_models = [m for m in effective_models if q in m.lower() or q in m.split("_")[0].lower()]

                # build and show figure (use normalized plotting dataframe)
                fig = make_cell_figure(df_plot, effective_models, effective_target, effective_metric, effective_k or None, effective_title, show_legend, [y_min, y_max] if (not pd.isna(y_min) and not pd.isna(y_max)) else None)
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{cell_idx}")

                # exports
                html, png = fig_to_downloads(fig)
                cols2 = st.columns([1, 1])
                with cols2[0]:
                    st.download_button(label="Download HTML", data=html, file_name=f"cell_{cell_idx+1}.html", mime="text/html")
                with cols2[1]:
                    if png is not None:
                        st.download_button(label="Download PNG", data=png, file_name=f"cell_{cell_idx+1}.png", mime="image/png")
                    else:
                        st.caption("PNG export requires kaleido")

            cell_idx += 1

st.sidebar.markdown("---")
st.sidebar.write("Quick stats")
st.sidebar.metric("Rows x Cols", f"{n_rows} x {n_cols}")
st.sidebar.metric("Total results", len(df))

