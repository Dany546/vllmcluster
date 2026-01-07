import json
import logging
import os
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from umap import UMAP  # GPU versions
from utils import dict_to_filename, get_lookups, table_exists

import wandb


def log_plot_plotly(
    df,
    umap_col=["umap_x", "umap_y"],
    tsne_col=["tsne_x", "tsne_y"],
    class_cols=None,  # single categorical column
    superclass_cols=None,  # single categorical column
    continuous_cols=None,  # single continuous column
    title="Embedding Visualization",
    colorscale="Viridis",
    subtitle: str = None,  # short text to show under the title (e.g., hyperparams)
    # Optional short labels and full params for hover
    umap_short: str = None,
    tsne_short: str = None,
    umap_params_full: str = None,
    tsne_params_full: str = None,
):
    """
    Interactive embedding visualization:
    - Two traces per subplot: grey background + foreground
    - Adaptive hover: class/superclass labels in categorical mode; value in continuous mode
    - Dropdown to switch encoding
    - Optional continuous range masking via opacity (buttons)
    """

    # =========================
    # CONFIG
    # =========================
    FIGURE_HEIGHT = 600
    FIGURE_WIDTH = 1200
    BACKGROUND_COLOR = "grey"
    BACKGROUND_OPACITY = 0.25
    FG_SIZE = 8
    BG_SIZE = 2

    # =========================
    # VALIDATION
    # =========================
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty")
    for col in umap_col + tsne_col:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # =========================
    # DATA PREP
    # =========================
    embeddings = [df[umap_col].values, df[tsne_col].values]
    emb_names = ["UMAP", "t-SNE"]
    n = len(df)

    # =========================
    # FIGURE
    # =========================
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=emb_names, horizontal_spacing=0.1
    )
    foreground_idxs = []

    # Background traces
    for idx, emb in enumerate(embeddings, 1):
        fig.add_trace(
            go.Scattergl(
                x=emb[:, 0],
                y=emb[:, 1],
                mode="markers",
                marker=dict(
                    size=BG_SIZE, color=BACKGROUND_COLOR, opacity=BACKGROUND_OPACITY
                ),
                name="Background",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=idx,
        )

    # Foreground traces (initial: categorical if available else continuous)
    def add_foreground(idx, emb, run_short=None, run_params=None):
        # Pretty-format params into HTML-friendly multi-line string
        def _format_params(p):
            if p is None:
                return None
            # If dict, pretty-print
            if isinstance(p, dict):
                pretty = json.dumps(p, indent=2, sort_keys=True)
            else:
                # try parse JSON string
                try:
                    parsed = json.loads(p)
                    pretty = json.dumps(parsed, indent=2, sort_keys=True)
                except Exception:
                    pretty = str(p)
            # convert newlines to <br> for HTML hover
            return pretty.replace("\n", "<br>")

        def _append_run_info(base_template, run_short, run_params):
            parts = []
            if run_short:
                parts.append(f"Run: {run_short}")
            params_html = _format_params(run_params)
            if params_html:
                parts.append("Params:")
                parts.append(params_html)
            if parts:
                # join with <br> for multiple lines
                return base_template.replace("<extra></extra>", "<br>" + "<br>".join(parts) + "<extra></extra>")
            return base_template

        if class_cols and class_cols in df.columns:
            # categorical init
            base = "x: %{x}<br>y: %{y}<br>Class: %{customdata}<extra></extra>"
            hover = _append_run_info(base, run_short, run_params)
            fig.add_trace(
                go.Scattergl(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    mode="markers",
                    marker=dict(size=FG_SIZE, color=df[class_col], opacity=0.9),
                    customdata=df[class_col],
                    hovertemplate=hover,
                    name=(run_short if run_short else "Classes"),
                    showlegend=True,
                ),
                row=1,
                col=idx,
            )
        elif superclass_col and superclass_col in df.columns:
            base = "x: %{x}<br>y: %{y}<br>Superclass: %{customdata}<extra></extra>"
            hover = _append_run_info(base, run_short, run_params)
            fig.add_trace(
                go.Scattergl(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    mode="markers",
                    marker=dict(size=FG_SIZE, color=df[superclass_col], opacity=0.9),
                    customdata=df[superclass_col],
                    hovertemplate=hover,
                    name=(run_short if run_short else "Superclasses"),
                    showlegend=True,
                ),
                row=1,
                col=idx,
            )
        elif continuous_col and continuous_col in df.columns:
            vals = df[continuous_col].values
            base = f"x: %{{x}}<br>y: %{{y}}<br>{continuous_col}: %{{customdata}}<extra></extra>"
            hover = _append_run_info(base, run_short, run_params)
            fig.add_trace(
                go.Scattergl(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    mode="markers",
                    marker=dict(
                        size=FG_SIZE,
                        color=vals,
                        colorscale=colorscale,
                        opacity=0.9,
                        showscale=(idx == 1),
                        colorbar=dict(title=continuous_col),
                        cmin=float(np.nanmin(vals)),
                        cmax=float(np.nanmax(vals)),
                    ),
                    customdata=vals,
                    hovertemplate=hover,
                    name=(run_short if run_short else continuous_col),
                    showlegend=True,
                ),
                row=1,
                col=idx,
            )
        else:
            base = "x: %{x}<br>y: %{y}<br>Idx: %{customdata}<extra></extra>"
            hover = _append_run_info(base, run_short, run_params)
            fig.add_trace(
                go.Scattergl(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    mode="markers",
                    marker=dict(size=FG_SIZE, color="steelblue", opacity=0.9),
                    customdata=np.arange(n),
                    hovertemplate=hover,
                    name=(run_short if run_short else "Points"),
                    showlegend=True,
                ),
                row=1,
                col=idx,
            )
        foreground_idxs.append(len(fig.data) - 1)

    for idx, emb in enumerate(embeddings, 1):
        add_foreground(idx, emb)

    # =========================
    # Dropdown buttons (encoding switch)
    # =========================
    buttons = []
    if class_col and class_col in df.columns:
        buttons.append(
            dict(
                label="Classes",
                method="update",
                args=[
                    {
                        "marker.color": [df[class_col]] * len(foreground_idxs),
                        "marker.colorscale": [None] * len(foreground_idxs),
                        "hovertemplate": [
                            "x: %{x}<br>y: %{y}<br>Class: %{customdata}<extra></extra>"
                        ]
                        * len(foreground_idxs),
                        "showlegend": [True] * len(foreground_idxs),
                    },
                    {"legend.title.text": "Classes"},
                ],
            )
        )
    if superclass_col and superclass_col in df.columns:
        buttons.append(
            dict(
                label="Superclasses",
                method="update",
                args=[
                    {
                        "marker.color": [df[superclass_col]] * len(foreground_idxs),
                        "marker.colorscale": [None] * len(foreground_idxs),
                        "hovertemplate": [
                            "x: %{x}<br>y: %{y}<br>Superclass: %{customdata}<extra></extra>"
                        ]
                        * len(foreground_idxs),
                        "showlegend": [True] * len(foreground_idxs),
                    },
                    {"legend.title.text": "Superclasses"},
                ],
            )
        )
    if continuous_col and continuous_col in df.columns:
        buttons.append(
            dict(
                label=continuous_col,
                method="update",
                args=[
                    {
                        "marker.color": [df[continuous_col]] * len(foreground_idxs),
                        "marker.colorscale": [colorscale] * len(foreground_idxs),
                        "hovertemplate": [
                            f"x: %{{x}}<br>y: %{{y}}<br>{continuous_col}: %{{customdata}}<extra></extra>"
                        ]
                        * len(foreground_idxs),
                        "showlegend": [False] * len(foreground_idxs),
                    },
                    {"legend.title.text": continuous_col},
                ],
            )
        )

    # =========================
    # Optional: continuous range masking via buttons
    # =========================
    if continuous_col and continuous_col in df.columns:
        vals = df[continuous_col].values
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        thresholds = np.linspace(vmin, vmax, 5)
        for t in thresholds:
            mask = (vals >= t) & (vals <= vmax)
            opacities = np.where(mask, 0.9, 0.1)
            buttons.append(
                dict(
                    label=f">= {t:.2f}",
                    method="restyle",
                    args=[
                        {"marker.opacity": [opacities] * len(foreground_idxs)},
                        foreground_idxs,
                    ],
                )
            )

    # =========================
    # Layout
    # =========================
    fig.update_layout(
        height=FIGURE_HEIGHT,
        width=FIGURE_WIDTH,
        hovermode="closest",
        title=title,
        plot_bgcolor="white",
        legend=dict(
            title="Classes"
            if (class_col and class_col in df.columns)
            else (
                "Superclasses"
                if (superclass_col and superclass_col in df.columns)
                else ""
            ),
            itemsizing="constant",
            yanchor="top",
            y=0.98,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
            )
        ],
        margin=dict(l=40, r=40, t=60, b=60),
    )

    # Optional subtitle (annotation) - useful to show hyperparameters used for UMAP/t-SNE
    if subtitle:
        # place just above the plot area, centered
        fig.add_annotation(
            dict(
                text=subtitle,
                x=0.5,
                y=1.03,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10, color="#444"),
                xanchor="center",
            )
        )

    # Optional: full params annotation + in-figure ℹ️ toggle (only if full params provided)
    if umap_params_full or tsne_params_full:
        def _pp_html(p):
            if p is None:
                return ""
            if isinstance(p, dict):
                pretty = json.dumps(p, indent=2, sort_keys=True)
            else:
                try:
                    pretty = json.dumps(json.loads(p), indent=2, sort_keys=True)
                except Exception:
                    pretty = str(p)
            return pretty.replace("\n", "<br>")

        umap_html = _pp_html(umap_params_full)
        tsne_html = _pp_html(tsne_params_full)
        full_html = f"<b>UMAP params</b><br>{umap_html}<br><b>t-SNE params</b><br>{tsne_html}"

        # add hidden annotation (will be toggled via updatemenu)
        fig.add_annotation(
            dict(
                text=full_html,
                x=0.99,
                y=1.03,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#ddd",
                borderwidth=1,
                font=dict(size=10, color="#222", family="monospace"),
                visible=False,
                xanchor="right",
            )
        )

        # compute index of the newly added annotation
        ann_idx = len(fig.layout.annotations) - 1

        # build show/hide buttons and append as a new updatemenu
        show_btn = dict(label="ℹ️ Params", method="relayout", args=[{f"annotations[{ann_idx}].visible": True}])
        hide_btn = dict(label="✖ Hide", method="relayout", args=[{f"annotations[{ann_idx}].visible": False}])

        # Preserve existing updatemenus and append new one
        existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
        existing.append(
            dict(
                buttons=[show_btn, hide_btn],
                direction="left",
                showactive=False,
                x=0.99,
                y=1.08,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=0),
            )
        )
        fig.update_layout(updatemenus=existing)

    return fig
