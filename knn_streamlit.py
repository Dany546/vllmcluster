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
# Compatibility shim: some Streamlit builds expose `experimental_rerun`,
# others rely on raising `RerunException` from the runtime. Ensure
# `st.experimental_rerun()` exists so the rest of the code can call it.
if not hasattr(st, "experimental_rerun"):
    try:
        from streamlit.runtime.scriptrunner import RerunException

        def _st_experimental_rerun() -> None:
            raise RerunException()

        st.experimental_rerun = _st_experimental_rerun
    except Exception:
        # Fallback: set a session flag and stop execution. This does not
        # perfectly mimic Streamlit's rerun behavior but avoids AttributeError.
        def _st_experimental_rerun() -> None:
            st.session_state["_rerun_requested"] = True
            try:
                st.stop()
            except Exception:
                # If st.stop() is not available for some reason, raise to abort.
                raise RuntimeError("Requested rerun but Streamlit runtime does not support it")

        st.experimental_rerun = _st_experimental_rerun
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import local utils robustly (avoid clashing with any installed 'utils' package)
try:
    from utils import parse_model_group, color_for_group
except Exception:
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("vllmcluster_utils", os.path.join(os.path.dirname(__file__), "utils.py"))
    vll_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vll_utils)
    parse_model_group = vll_utils.parse_model_group
    color_for_group = vll_utils.color_for_group


def get_db_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--local", action="store_true")
    # Use parse_known_args to avoid SystemExit when Streamlit passes its own CLI flags
    args, _ = parser.parse_known_args()

    if args.db_path:
        return args.db_path
    if args.local:
        local_path = os.path.expanduser("~/knn_results.db")
        return local_path
    return "/CECI/home/ucl/irec/darimez/knn_results.db"


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
                     y_range: Optional[List[float]],
                     plot_style: str = "box",
                     box_gap: float = 0.05,
                     box_group_gap: float = 0.02,
                     marker_size: int = 6,
                     include_distance_in_group: bool = False,
                     boxpoints: str = "outliers",
                     notched: bool = False,
                     quartilemethod: str = "linear",
                     highlight_fliers: bool = False,
                     show_whisker_endpoints: bool = False):
    """Create a Plotly figure for the given cell with grouping (hue-like) support.

    New features:
    - Use `offsetgroup` and `legendgroup` so traces that share a group are offset (like hue).
    - Deterministic color mapping per group via `color_for_group`.
    - Optional `boxpoints` and `notched` box options for clarity.
    """
    fig = go.Figure()
    if df.empty or not models:
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    # ensure k_values is list
    if k_values:
        df = df[df["k"].isin(k_values)]

    # If no data after filtering, return a message
    df_target = df[df["target"] == target]
    if df_target.empty:
        fig.add_annotation(text="No data for this target", showarrow=False)
        return fig

    # For categorical placement we map k values to category strings
    unique_k = sorted(df["k"].unique())
    k_categories = [str(k) for k in unique_k]

    # Only consider models that actually have data for this target
    active_models = [m for m in models if not df_target[df_target["model_full"] == m].empty]
    if not active_models:
        fig.add_annotation(text="No selected models with data", showarrow=False)
        return fig

    n_active = len(active_models)
    # group total width (in x units) per k category; boxes will be placed within [-group_w/2, +group_w/2]
    group_total_width = min(0.8, 0.8)
    offset_step = group_total_width / max(1, n_active)
    width = max(min(offset_step * 0.8, 0.6), 0.02)

    # numeric mapping for k categories to support offsets
    k_to_idx = {k: idx for idx, k in enumerate(unique_k)}

    for i, model in enumerate(active_models):
        sub = df_target[df_target["model_full"] == model]

        base, suffix = parse_model_group(model)
        group_key = model if include_distance_in_group else base
        color = color_for_group(group_key)

        if plot_style == "box":
            # numeric x positions with small offsets so boxes are side-by-side
            offset = (i - (n_active - 1) / 2.0) * offset_step
            x = sub["k"].map(k_to_idx).astype(float) + offset
            try:
                fig.add_trace(
                    go.Box(
                        x=x,
                        y=sub[metric],
                        name=(model if show_distance_metric else base),
                        marker_color=color,
                        boxmean=True,
                        visible=True,
                        width=width,
                        legendgroup=str(group_key),  # group in legend by base model
                        boxpoints=(boxpoints if boxpoints != "False" else False),
                        notched=notched,
                        quartilemethod=quartilemethod,
                    )
                )
            except Exception:
                # Fallback: older plotly versions may not support some attributes; fall back to non-offset x
                x_fallback = sub["k"].astype(str)
                fig.add_trace(
                    go.Box(
                        x=x_fallback,
                        y=sub[metric],
                        name=(model if show_distance_metric else base),
                        marker_color=color,
                        boxmean=True,
                        visible=True,
                        width=width,
                    )
                )

            # Optionally highlight fliers (outliers) explicitly with scatter markers for clarity
            try:
                vals = sub[metric].astype(float)
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                low_end = q1 - 1.5 * iqr
                high_end = q3 + 1.5 * iqr
                outlier_mask = (vals < low_end) | (vals > high_end)
                whisker_low_val = vals[~outlier_mask].min() if (~outlier_mask).any() else vals.min()
                whisker_high_val = vals[~outlier_mask].max() if (~outlier_mask).any() else vals.max()

                if highlight_fliers and outlier_mask.any():
                    # scatter the outliers so they are visible and placed at numeric x positions
                    x_out = sub["k"].map(k_to_idx).astype(float)[outlier_mask] + offset
                    fig.add_trace(
                        go.Scatter(
                            x=x_out,
                            y=vals[outlier_mask],
                            mode='markers',
                            marker=dict(color=color, size=max(4, marker_size - 1), symbol='circle-open'),
                            name=None,
                            showlegend=False,
                            hoverinfo='y',
                        )
                    )

                if show_whisker_endpoints:
                    # add small markers at whisker endpoints for each k (with same offset)
                    xs = []
                    ys = []
                    ks = sorted(sub["k"].unique())
                    for k in ks:
                        s_k = sub[sub["k"] == k][metric].astype(float)
                        if s_k.empty:
                            continue
                        q1_k = s_k.quantile(0.25)
                        q3_k = s_k.quantile(0.75)
                        iqr_k = q3_k - q1_k
                        low_k = q1_k - 1.5 * iqr_k
                        high_k = q3_k + 1.5 * iqr_k
                        mask_k = (s_k < low_k) | (s_k > high_k)
                        wl = s_k[~mask_k].min() if (~mask_k).any() else s_k.min()
                        wh = s_k[~mask_k].max() if (~mask_k).any() else s_k.max()
                        xs += [k_to_idx[k] + offset, k_to_idx[k] + offset]
                        ys += [wl, wh]
                    if xs:
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode='markers',
                                marker=dict(color=color, size=6, symbol='diamond'),
                                name=None,
                                showlegend=False,
                                hoverinfo='y',
                            )
                        )
            except Exception:
                pass
        else:  # scatter
            # map k to sequential numeric positions and add a small deterministic offset per model
            k_to_idx = {k: idx for idx, k in enumerate(unique_k)}
            base_x = sub["k"].map(k_to_idx).astype(float)
            # offset traces so models do not perfectly overlap; deterministic offset based on i
            offset = (i - (n_active - 1) / 2.0) * 0.08
            rng = np.random.default_rng(42 + i)
            jitter = rng.normal(0, 0.02, size=len(base_x))
            x_pos = base_x + offset + jitter
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=sub[metric],
                    mode='markers',
                    name=(model if show_distance_metric else base),
                    marker=dict(color=color, size=marker_size),
                    visible=True,
                    legendgroup=str(group_key),
                )
            )

    # Layout adjustments
    xaxis = dict(title="k neighbors")
    if plot_style == "box":
        # numeric ticks for box plots with offsets: show k values at integer positions
        xaxis.update(
            tickmode='array',
            tickvals=list(range(len(k_categories))),
            ticktext=k_categories,
            range=[-0.6, len(k_categories) - 1 + 0.6],
        )
    else:
        # numeric ticks for scatter using index positions but show labels as original k values
        xaxis.update(
            tickmode='array',
            tickvals=list(range(len(k_categories))),
            ticktext=k_categories,
        )

    fig.update_layout(
        title=title or f"{metric.upper()} - {target}",
        xaxis=xaxis,
        yaxis_title=metric,
        template="plotly_white",
        boxmode="group",
        height=420,
        showlegend=show_legend,
    )

    # tune box spacing if requested (no effect on scatter)
    try:
        fig.update_layout(boxgap=box_gap, boxgroupgap=box_group_gap)
    except Exception:
        pass

    if y_range and len(y_range) == 2:
        fig.update_yaxes(range=y_range)

    return fig


def fig_to_downloads(fig: go.Figure):
    # sanitize any invalid axis ranges (NaN / infinite) that can break the
    # frontend when Plotly serializes the figure
    try:
        import math
        layout_json = fig.layout.to_plotly_json()
        for key, val in layout_json.items():
            # keys like 'yaxis', 'yaxis2', etc.
            if key.startswith("yaxis") and isinstance(val, dict):
                r = val.get("range")
                if r and isinstance(r, (list, tuple)):
                    bad = False
                    for v in r:
                        try:
                            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                                bad = True
                                break
                        except Exception:
                            bad = True
                            break
                    if bad:
                        # clear ranges for all y axes to let Plotly autoscale
                        try:
                            fig.update_yaxes(range=None)
                        except Exception:
                            pass
                        break
    except Exception:
        pass

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


@st.cache_data
def load_proj_metadata(proj_dir: str = "/CECI/home/ucl/irec/darimez/proj") -> pd.DataFrame:
    """Load projection run metadata (umap/tsne) if available.

    Returns a DataFrame with at least columns: run_id, model, algo, n_components (when present) and any extra params.
    """
    runs = []
    for algo in ["umap", "tsne"]:
        dbfile = os.path.join(proj_dir, f"{algo}.db")
        if not os.path.exists(dbfile):
            continue
        try:
            conn = sqlite3.connect(dbfile)
            dfm = pd.read_sql("SELECT * FROM metadata", conn)
            conn.close()
            if dfm.empty:
                continue
            dfm["algo"] = algo
            # normalize column for n_components if present
            if "n_components" in dfm.columns:
                # ensure numeric
                try:
                    dfm["n_components"] = pd.to_numeric(dfm["n_components"], errors="coerce")
                except Exception:
                    pass
            runs.append(dfm)
        except Exception:
            continue
    if not runs:
        return pd.DataFrame()
    return pd.concat(runs, ignore_index=True)


proj_meta = load_proj_metadata()


st.sidebar.header("Layout & Global Filters")
n_rows = st.sidebar.number_input("Grid rows", min_value=1, max_value=4, value=1)
n_cols = st.sidebar.number_input("Grid cols", min_value=1, max_value=4, value=2)

# Page selector: Grid or Stats
page = st.sidebar.radio("Page", options=["Grid", "Statistics"], index=0)

# Global k options
k_options = sorted(df["k"].unique().tolist())
global_k = st.sidebar.multiselect("Global k values", options=k_options, default=k_options)

# Projection-based N-components filter (if metadata found)
if not proj_meta.empty:
    ncomp_options = sorted(proj_meta["n_components"].dropna().unique().astype(int).tolist())
else:
    ncomp_options = []

include_unknown_ncomp = st.sidebar.checkbox("Include runs without n_components metadata", value=True)
selected_ncomps = st.sidebar.multiselect("Filter runs by n_components", options=ncomp_options, default=ncomp_options if ncomp_options else [])
                

# Available models (apply Ncomp filter when selected)
models_all = sorted(df["model_full"].unique().tolist())
if selected_ncomps and not proj_meta.empty:
    def _get_model_ncomp(model_name: str):
        # try exact match then substring match
        m = proj_meta.loc[proj_meta["run_id"] == model_name]
        if m.empty:
            m = proj_meta.loc[proj_meta["run_id"].apply(lambda r: r in str(model_name) or str(model_name) in str(r))]
        if m.empty:
            return None
        return m.iloc[0].get("n_components", None)

    filtered = []
    for m in models_all:
        n = _get_model_ncomp(m)
        if n is None:
            if include_unknown_ncomp:
                filtered.append(m)
        else:
            try:
                if int(n) in selected_ncomps:
                    filtered.append(m)
            except Exception:
                # keep if parsing fails and user opted to include unknowns
                if include_unknown_ncomp:
                    filtered.append(m)
    models_all = filtered
else:
    models_all = sorted(df["model_full"].unique().tolist())

# Session-managed global model selection so buttons can update it
if "global_models" not in st.session_state:
    st.session_state["global_models"] = models_all.copy()

# Quick model controls
if st.sidebar.button("Select all models"):
    st.session_state["global_models"] = models_all.copy()
    st.session_state["global_models_widget"] = models_all.copy()
    st.experimental_rerun()
if st.sidebar.button("Clear models"):
    st.session_state["global_models"] = []
    st.session_state["global_models_widget"] = []
    st.experimental_rerun()

search_term = st.sidebar.text_input("Select models matching (substring)", value="", key="model_search")
if st.sidebar.button("Select matching") and search_term:
    matches = [m for m in models_all if search_term.lower() in m.lower()]
    st.session_state["global_models"] = matches
    st.session_state["global_models_widget"] = matches
    st.experimental_rerun()

# Grouping controls: allow grouping by base model or full name
# Default: do NOT group runs automatically; user must opt in
if "group_by_base" not in st.session_state:
    st.session_state["group_by_base"] = False

group_by_base = st.sidebar.checkbox(
    "Group runs by base model (merge distance-metric variants)",
    value=st.session_state["group_by_base"],
    key="group_by_base",
    help="When enabled, runs like 'resnet50_l2' and 'resnet50_cos' will be shown as a single group 'resnet50'.",
)

# Quick actions for grouping
if st.sidebar.button("Separate all runs"):
    st.session_state["group_by_base"] = False
    st.session_state["global_models"] = models_all.copy()
    st.session_state["global_models_widget"] = models_all.copy()
    st.experimental_rerun()
if st.sidebar.button("Group runs by base model"):
    st.session_state["group_by_base"] = True
    base_groups = sorted({parse_model_group(m)[0] for m in models_all})
    st.session_state["global_model_groups"] = base_groups.copy()
    expanded = [m for m in models_all if parse_model_group(m)[0] in st.session_state["global_model_groups"]]
    st.session_state["global_models"] = expanded
    st.session_state["global_models_widget"] = expanded
    st.experimental_rerun()

# Show whether to include distance metric in display labels (affects titles / legends)
show_distance_metric = st.sidebar.checkbox("Include distance metric in labels", value=False)

# (Global models multiselect will be created after group-selection widgets)

# Show groups multiselect to select groups instead of individual models (optional)
base_groups = sorted({parse_model_group(m)[0] for m in models_all})
if group_by_base:
    # default groups selected is all groups
    if "global_model_groups" not in st.session_state:
        st.session_state["global_model_groups"] = base_groups.copy()
    selected_groups = st.sidebar.multiselect("Select model groups", options=base_groups, default=st.session_state["global_model_groups"], key="global_model_groups")
    if st.sidebar.button("Apply groups"):
        # expand selected groups into concrete models
        expanded = [m for m in models_all if parse_model_group(m)[0] in selected_groups]
        st.session_state["global_models"] = expanded
        st.session_state["global_models_widget"] = expanded
st.sidebar.write("Model groups (color)")
current_groups = sorted({parse_model_group(m)[0] if group_by_base else m for m in st.session_state["global_models"]})
cols = st.sidebar.columns(2)
for i, g in enumerate(current_groups):
    color = color_for_group(g)
    st.sidebar.markdown(f"- <span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border-radius:2px;'></span> `{g}`", unsafe_allow_html=True)

st.sidebar.markdown("---")
# Show n_components / proj hyperparams for selected runs (if available)
if not proj_meta.empty:
    sel_models = st.session_state.get("global_models", [])
    meta_rows = []
    for m in sel_models:
        match = proj_meta.loc[proj_meta["run_id"] == m]
        if match.empty:
            # try substring heuristic
            match = proj_meta.loc[proj_meta["run_id"].apply(lambda r: r in str(m) or str(m) in str(r))]
        if match.empty:
            meta_rows.append({"model": m, "n_components": None, "algo": None})
        else:
            row = match.iloc[0]
            ncomp = row.get("n_components", None)
            try:
                ncomp = int(ncomp) if pd.notnull(ncomp) else None
            except Exception:
                pass
            meta_rows.append({"model": m, "n_components": ncomp, "algo": row.get("algo", None)})
    if meta_rows:
        st.sidebar.write("Selected runs metadata (projection)")
        st.sidebar.dataframe(pd.DataFrame(meta_rows), use_container_width=True)

    # --- Advanced hyperparameter run filter ---
    st.sidebar.markdown("### Advanced run filter")
    # n_components filter (if projection metadata is available)
    ncomp_choice = []
    if not proj_meta.empty:
        ncomp_choice = st.sidebar.multiselect("n_components (projection)", options=ncomp_options, default=[])

    # Projection / table filter (derived from model_full format '<run>.<proj>.<id>')
    proj_options = sorted({m.split('.')[1] for m in models_all if '.' in m})
    proj_choice = st.sidebar.multiselect("Projection / table", options=proj_options, default=[])

    # Model substring filter
    model_pattern = st.sidebar.text_input("Model substring (case-insensitive)", value="")

    # Combine mode: AND (all) vs OR (any)
    combine_or = st.sidebar.checkbox("Combine filters with OR (match any)", value=False)

    if st.sidebar.button("Apply hyperparameter filter"):
        def _resolve_ncomp(mname: str):
            if proj_meta.empty:
                return None
            m = proj_meta.loc[proj_meta["run_id"] == mname]
            if m.empty:
                m = proj_meta.loc[proj_meta["run_id"].apply(lambda r: r in str(mname) or str(mname) in str(r))]
            if m.empty:
                return None
            try:
                val = m.iloc[0].get("n_components", None)
                return int(val) if pd.notnull(val) else None
            except Exception:
                return None

        matches = []
        for m in models_all:
            checks = []
            if ncomp_choice:
                nval = _resolve_ncomp(m)
                checks.append(nval in ncomp_choice)
            if proj_choice:
                proj_name = m.split('.')[1] if '.' in m else ''
                checks.append(proj_name in proj_choice)
            if model_pattern:
                checks.append(model_pattern.lower() in m.lower())

            if not checks:
                continue
            if combine_or:
                if any(checks):
                    matches.append(m)
            else:
                if all(checks):
                    matches.append(m)

        if matches:
            st.session_state["global_models"] = matches
            st.session_state["global_models_widget"] = matches
            st.experimental_rerun()

    if st.sidebar.button("Clear hyperparameter filters"):
        st.session_state["global_models"] = models_all.copy()
        st.session_state["global_models_widget"] = models_all.copy()
        st.experimental_rerun()

    # Global models multiselect (placed after filters so programmatic writes succeed)
    global_models = st.sidebar.multiselect(
        "Global models",
        options=models_all,
        default=st.session_state.get("global_models", models_all.copy()),
        key="global_models_widget",
    )
    # Keep session state value in sync with widget selection
    if st.session_state.get("global_models") != st.session_state.get("global_models_widget"):
        st.session_state["global_models"] = st.session_state["global_models_widget"]

st.sidebar.write("Per-cell fallbacks (cells override when non-empty)")
global_metric = st.sidebar.selectbox("Global metric", options=["correlation/ARI", "mae/accuracy", "r2", "error_dist_corr"], index=0)
global_target = st.sidebar.selectbox("Global target", options=sorted(df["target"].unique().tolist()), index=0)
global_query = st.sidebar.text_input("Global trace filter (substring)", value="", help="Case-insensitive substring to filter traces")

# Plot style & spacing controls (help keep boxes compact when k values are sparse)
global_plot_style = st.sidebar.selectbox("Default plot style", options=["box", "scatter"], index=0)
box_gap = st.sidebar.slider("Box gap (between boxes)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, help="Gap between boxes of adjacent k values; smaller -> tighter")
box_group_gap = st.sidebar.slider("Box group gap (between groups)", min_value=0.0, max_value=0.5, value=0.02, step=0.01, help="Gap between groups of boxes (per model)")
marker_size = st.sidebar.slider("Scatter marker size", min_value=2, max_value=20, value=6, step=1)

# Box visual options
boxpoints = st.sidebar.selectbox("Box points", options=["outliers", "all", "suspectedoutliers", "False"], index=0)
notched = st.sidebar.checkbox("Notched boxes", value=False)
quartilemethod = st.sidebar.selectbox("Quartile method", options=["linear", "inclusive", "exclusive"], index=0)
highlight_fliers = st.sidebar.checkbox("Highlight fliers (explicit points)", value=True)
show_whisker_endpoints = st.sidebar.checkbox("Show whisker endpoints", value=False)

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
            "correlation/ARI": ["mean", "std"],
                "mae/accuracy": ["mean", "std"],
                "error_dist_corr": ["mean", "std"],
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
                        options=["(use global)", "correlation/ARI", "mae/accuracy", "r2", "error_dist_corr"],
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
                    chosen_plot = st.selectbox(f"Plot style (cell {cell_idx+1})", options=["(use global)", "box", "scatter"], index=0, key=f"plot_{cell_idx}")
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

                # determine effective plot style (per-cell override -> global)
                effective_plot = chosen_plot if chosen_plot != "(use global)" else global_plot_style

                # build and show figure
                fig = make_cell_figure(
                    df,
                    effective_models,
                    effective_target,
                    effective_metric,
                    effective_k or None,
                    effective_title,
                    show_legend,
                    [y_min, y_max] if (not pd.isna(y_min) and not pd.isna(y_max)) else None,
                    plot_style=effective_plot,
                    box_gap=box_gap,
                    box_group_gap=box_group_gap,
                    marker_size=marker_size,
                    include_distance_in_group=(not group_by_base),
                    boxpoints=boxpoints,
                    notched=notched,
                    quartilemethod=quartilemethod,
                    highlight_fliers=highlight_fliers,
                    show_whisker_endpoints=show_whisker_endpoints,
                )
                # Provide a unique key per cell to avoid StreamlitDuplicateElementId
                chart_key = f"knn_figure_cell_{cell_idx}"
                try:
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                except Exception as e:
                    # In rare cases Streamlit may still complain about duplicate IDs; fall back to a uuid-based key
                    import uuid
                    st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_{uuid.uuid4().hex}")

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

