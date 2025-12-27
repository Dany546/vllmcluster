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


def get_db_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    if args.db_path:
        return args.db_path
    if args.local:
        local_path = os.path.expanduser("~/knn_results.db")
        return local_path
    return "/globalscratch/ucl/irec/darimez/dino/knn_results.db"


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

# Load DB
db_path = get_db_path()
df = load_knn_results(db_path)
if df.empty:
    st.error("No data loaded from database")
    st.stop()


st.sidebar.header("Layout & Global Filters")
n_rows = st.sidebar.number_input("Grid rows", min_value=1, max_value=4, value=1)
n_cols = st.sidebar.number_input("Grid cols", min_value=1, max_value=4, value=2)

# Page selector: Grid or Stats
page = st.sidebar.radio("Page", options=["Grid", "Statistics"], index=0)

# Global k options
k_options = sorted(df["k"].unique().tolist())
global_k = st.sidebar.multiselect("Global k values", options=k_options, default=k_options)

# Available models
models_all = sorted(df["model_full"].unique().tolist())
global_models = st.sidebar.multiselect("Global models", options=models_all, default=models_all)

st.sidebar.markdown("---")
st.sidebar.write("Per-cell fallbacks (cells override when non-empty)")
global_metric = st.sidebar.selectbox("Global metric", options=["corr", "mae", "r2"], index=0)
global_target = st.sidebar.selectbox("Global target", options=sorted(df["target"].unique().tolist()), index=0)
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
    df_summary = df.copy()
    if global_k:
        df_summary = df_summary[df_summary["k"].isin(global_k)]
    if global_models:
        df_summary = df_summary[df_summary["model_full"].isin(global_models)]
    if global_query:
        q = global_query.lower()
        df_summary = df_summary[df_summary["model_full"].str.lower().str.contains(q) | df_summary["model"].str.lower().str.contains(q)]

    st.write("### Data Summary")
    st.dataframe(
        df_summary.groupby(["model", "target", "k"]).agg({
            "corr": ["mean", "std"],
            "mae": ["mean", "std"],
            "r2": ["mean", "std"],
        }).round(4),
        use_container_width=True
    )

else:
    st.write(f"Showing grid: {n_rows} x {n_cols}")
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
                        options=["(use global)"] + sorted(df["target"].unique().tolist()),
                        index=0,
                        key=f"target_{cell_idx}"
                    )

                    chosen_metric = st.selectbox(
                        f"Metric (cell {cell_idx+1})",
                        options=["(use global)", "corr", "mae", "r2"],
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

                # build and show figure
                fig = make_cell_figure(df, effective_models, effective_target, effective_metric, effective_k or None, effective_title, show_legend, [y_min, y_max] if (not pd.isna(y_min) and not pd.isna(y_max)) else None)
                st.plotly_chart(fig, use_container_width=True)

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

