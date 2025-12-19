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


def get_colormap(n, palette="husl"):
    """
    Return a list of n distinct colors.
    Uses Plotly qualitative palettes.
    """
    if palette.lower() in ["husl", "set2", "pastel", "bold"]:
        base = (
            px.colors.qualitative.Set2
            if palette.lower() == "set2"
            else px.colors.qualitative.Bold
        )
    elif palette.lower() == "set3":
        base = px.colors.qualitative.Set3
    else:
        base = px.colors.qualitative.Dark24
    colors = []
    while len(colors) < n:
        colors.extend(base)
    return colors[:n]


def log_plot_plotly(
    df,
    umap_col=["umap_x", "umap_y"],
    tsne_col=["tsne_x", "tsne_y"],
    class_cols=None,
    superclass_cols=None,
    continuous_cols=None,
    title="Embedding Visualization",
):
    """
    Creates interactive embedding visualization with togglable legend.

    Key behavior:
    - When a class is toggled OFF via legend: shows points in GREY with SMALL size
    - When a class is toggled ON via legend: shows points in COLOR with NORMAL size
    - Background layer always visible in light grey

    Parameters:
    -----------
    df : DataFrame
        Data containing embeddings and class/continuous variables
    umap_col : list
        Column names for UMAP coordinates [x, y]
    tsne_col : list
        Column names for t-SNE coordinates [x, y]
    class_cols : list, optional
        Column names for one-hot encoded classes
    superclass_cols : list, optional
        Column names for one-hot encoded superclasses
    continuous_cols : list, optional
        Column names for continuous variables (e.g., ["error", "loss"])
    title : str
        Plot title
    run : wandb.run, optional
        Wandb run object for logging

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """

    # ============================================
    # CONFIGURATION - Modify these to customize appearance
    # ============================================
    FIGURE_HEIGHT = 600
    FIGURE_WIDTH = 1200

    # Point sizes
    ACTIVE_POINT_SIZE = 8  # Size when class is toggled ON
    INACTIVE_POINT_SIZE = 4  # Size when class is toggled OFF
    BACKGROUND_POINT_SIZE = 2  # Size for permanent background

    # Opacity
    ACTIVE_OPACITY = 0.9  # Opacity when toggled ON
    INACTIVE_OPACITY = 0.3  # Opacity when toggled OFF (grey)
    BACKGROUND_OPACITY = 0.25  # Opacity for permanent background

    # Colors
    INACTIVE_COLOR = "grey"  # Color when toggled OFF
    BACKGROUND_COLOR = "grey"

    # ============================================
    # VALIDATION
    # ============================================
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty")

    for col in umap_col + tsne_col:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # ============================================
    # DATA PREPARATION
    # ============================================
    umap_embedding = df[umap_col].values
    tsne_embedding = df[tsne_col].values
    embeddings = [umap_embedding, tsne_embedding]
    emb_names = ["UMAP", "t-SNE"]

    norm_sizes_active = np.full(len(df), ACTIVE_POINT_SIZE)
    norm_sizes_inactive = np.full(len(df), INACTIVE_POINT_SIZE)
    norm_sizes_bg = np.full(len(df), BACKGROUND_POINT_SIZE)

    # ============================================
    # FIGURE CREATION
    # ============================================
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=emb_names, horizontal_spacing=0.1
    )

    # Track trace organization
    background_idxs = []
    encoding_groups = {}

    # ============================================
    # ALWAYS-VISIBLE BACKGROUND LAYER
    # Shows all points in grey - never hidden
    # ============================================
    for idx, embedding in enumerate(embeddings, 1):
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=norm_sizes_bg,
                    color=BACKGROUND_COLOR,
                    opacity=BACKGROUND_OPACITY,
                    line=dict(width=0),
                ),
                name="Background",
                showlegend=False,
                hoverinfo="skip",
                visible=True,  # Always visible
            ),
            row=1,
            col=idx,
        )
        background_idxs.append(len(fig.data) - 1)

    # ============================================
    # CLASSES - Categorical with toggle behavior
    # Each class gets TWO traces per subplot:
    #   1. Active trace (colored, normal size) - visible by default
    #   2. Inactive trace (grey, small size) - hidden by default
    # Plotly's legend groups them so clicking toggles between active/inactive
    # ============================================
    if class_cols is not None:
        if not all(col in df.columns for col in class_cols):
            raise ValueError("Some class columns not found in DataFrame")

        class_matrix = df[class_cols].values
        classes = np.argmax(class_matrix, axis=1)
        class_names = class_cols
        class_colors = get_colormap(len(class_names), "husl")
        encoding_groups["classes"] = []

        for idx, embedding in enumerate(embeddings, 1):
            for cat_idx, cat_name in enumerate(class_names):
                mask = classes == cat_idx

                # ACTIVE trace (colored, shown when toggled ON)
                fig.add_trace(
                    go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode="markers",
                        marker=dict(
                            size=norm_sizes_active[mask],
                            color=class_colors[cat_idx],
                            opacity=ACTIVE_OPACITY,
                            line=dict(width=0.5, color="white"),
                        ),
                        name=cat_name,
                        legendgroup=cat_name,  # Groups active/inactive together
                        showlegend=(idx == 1),  # Show legend only for first subplot
                        visible=True,  # Initially active
                    ),
                    row=1,
                    col=idx,
                )
                encoding_groups["classes"].append(len(fig.data) - 1)

                # INACTIVE trace (grey, shown when toggled OFF)
                # fig.add_trace(
                #     go.Scatter(
                #         x=embedding[mask, 0],
                #         y=embedding[mask, 1],
                #         mode="markers",
                #         marker=dict(
                #             size=norm_sizes_inactive[mask],
                #             color=INACTIVE_COLOR,
                #             opacity=INACTIVE_OPACITY,
                #             line=dict(width=0),
                #         ),
                #         name=cat_name,
                #         legendgroup=cat_name,  # Same group as active
                #         showlegend=False,  # Don't show in legend (controlled by active trace)
                #         visible="legendonly",  # Hidden until active is toggled off
                #     ),
                #     row=1,
                #     col=idx,
                # )
                # encoding_groups["classes"].append(len(fig.data) - 1)

    # ============================================
    # SUPERCLASSES - Same dual-trace pattern as classes
    # ============================================
    if superclass_cols is not None:
        if not all(col in df.columns for col in superclass_cols):
            raise ValueError("Some superclass columns not found in DataFrame")

        superclass_matrix = df[superclass_cols].values
        superclasses = np.argmax(superclass_matrix, axis=1)
        superclass_names = superclass_cols
        superclass_colors = get_colormap(len(superclass_names), "Set3")
        encoding_groups["superclasses"] = []

        for idx, embedding in enumerate(embeddings, 1):
            for cat_idx, cat_name in enumerate(superclass_names):
                mask = superclasses == cat_idx

                # ACTIVE trace
                fig.add_trace(
                    go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode="markers",
                        marker=dict(
                            size=norm_sizes_active[mask],
                            color=superclass_colors[cat_idx],
                            opacity=ACTIVE_OPACITY,
                            line=dict(width=0.5, color="white"),
                        ),
                        name=cat_name,
                        legendgroup=cat_name,
                        showlegend=(idx == 1),
                        visible=False,  # Hidden initially (classes shown by default)
                    ),
                    row=1,
                    col=idx,
                )
                encoding_groups["superclasses"].append(len(fig.data) - 1)

                # INACTIVE trace
                # fig.add_trace(
                #     go.Scatter(
                #         x=embedding[mask, 0],
                #         y=embedding[mask, 1],
                #         mode="markers",
                #         marker=dict(
                #             size=norm_sizes_inactive[mask],
                #             color=INACTIVE_COLOR,
                #             opacity=INACTIVE_OPACITY,
                #             line=dict(width=0),
                #         ),
                #         name=cat_name,
                #         legendgroup=cat_name,
                #         showlegend=False,
                #         visible=False,
                #     ),
                #     row=1,
                #     col=idx,
                # )
                # encoding_groups["superclasses"].append(len(fig.data) - 1)

    # ============================================
    # CONTINUOUS VARIABLES - Single colormap trace per variable
    # No toggling behavior - shows all points with continuous color scale
    # ============================================
    if continuous_cols is not None:
        for col in continuous_cols:
            if col not in df.columns:
                raise ValueError(f"Continuous column '{col}' not found in DataFrame")

            values = df[col].values
            encoding_groups[col] = []

            for idx, embedding in enumerate(embeddings, 1):
                fig.add_trace(
                    go.Scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        mode="markers",
                        marker=dict(
                            size=norm_sizes_active,
                            color=values,
                            colorscale="Viridis",
                            opacity=ACTIVE_OPACITY,
                            line=dict(width=0.5, color="white"),
                            showscale=(idx == 1),  # Show colorbar on first subplot
                            colorbar=dict(title=col),
                            cmin=float(np.nanmin(values)),
                            cmax=float(np.nanmax(values)),
                        ),
                        name=col,
                        showlegend=False,
                        visible=False,  # Hidden initially
                    ),
                    row=1,
                    col=idx,
                )
                encoding_groups[col].append(len(fig.data) - 1)

    # ============================================
    # DROPDOWN MENU - Switch between encoding types
    # Each button shows/hides appropriate traces and controls legend visibility
    # ============================================
    total_traces = len(fig.data)

    def button_for(group_name, showlegend_on=True):
        """
        Creates a dropdown button that shows only the selected encoding type.

        Parameters:
        -----------
        group_name : str
            Name of encoding group (e.g., "classes", "superclasses", "error")
        showlegend_on : bool
            Whether to show legend for this encoding type
        """
        visible = [False] * total_traces

        # Background always visible
        for bi in background_idxs:
            visible[bi] = True

        # Show selected encoding group
        for gi in encoding_groups.get(group_name, []):
            visible[gi] = True

        # Update legend title based on encoding type
        if showlegend_on:
            legend_title = f"{group_name.replace('_', ' ').title()}"
        else:
            legend_title = ""

        return dict(
            label=group_name.replace("_", " ").title(),
            method="update",
            args=[
                {"visible": visible},
                {"showlegend": showlegend_on, "legend.title.text": legend_title},
            ],
        )

    def view_or_hide_all_button(group_name, hide=False):
        """
        Creates a dropdown button that shows only the selected encoding type.

        Parameters:
        -----------
        group_name : str
            Name of encoding group (e.g., "classes", "superclasses", "error")
        showlegend_on : bool
            Whether to show legend for this encoding type
        """
        visible = [False] * total_traces

        # Background always visible
        for bi in background_idxs:
            visible[bi] = True

        fig.update_menus[0].update(
            {"visible": visible},
            {"showlegend": showlegend_on, "legend.title.text": legend_title},
        )

        # Show selected encoding group
        for gi in encoding_groups.get(group_name, []):
            visible[gi] = not hide

        return dict(
            label="View all" if not hide else "Hide all",
            method="update",
            args=[
                {"visible": visible},
            ],
        )

    # Determine default encoding (prefer classes if available)
    default_group = None
    if "classes" in encoding_groups:
        default_group = "classes"
    elif "superclasses" in encoding_groups:
        default_group = "superclasses"
    else:
        default_group = next(iter(encoding_groups), None)

    # Build dropdown buttons
    color_buttons = [[]]
    for name in encoding_groups.keys():
        # Show legend for categorical (classes/superclasses), hide for continuous
        showlegend_on = name in ["classes", "superclasses"]
        color_buttons[0].append(button_for(name, showlegend_on=showlegend_on))

    # color_buttons.append([])
    # for name in encoding_groups.keys():
    #     color_buttons[1].append(view_or_hide_all_button(name, hide=False))
    # color_buttons.append([])
    # for name in encoding_groups.keys():
    #     color_buttons[2].append(view_or_hide_all_button(name, hide=True))

    # Apply default visibility
    initial_visible = [False] * total_traces
    for bi in background_idxs:
        initial_visible[bi] = True
    if default_group is not None:
        for gi in encoding_groups[default_group]:
            initial_visible[gi] = True

    for i, v in enumerate(initial_visible):
        fig.data[i].visible = v

    # Set initial legend title
    initial_legend_title = ""
    if default_group in ["classes", "superclasses"]:
        initial_legend_title = f"{default_group.replace('_', ' ').title()}"

    # ============================================
    # LAYOUT CONFIGURATION
    # ============================================
    fig.update_layout(
        height=FIGURE_HEIGHT,
        width=FIGURE_WIDTH,
        hovermode="closest",
        title=title,
        plot_bgcolor="white",  # inside plotting area
        legend=dict(
            title=initial_legend_title,
            itemsizing="constant",
            yanchor="top",
            y=0.98,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        updatemenus=[
            dict(
                buttons=color_buttons[0],
                direction="up",
                showactive=True,
                x=0.05,
                y=0.05,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=0, b=10),
            ),
            # dict(
            #     buttons=color_buttons[1],
            #     direction="up",
            #     showactive=True,
            #     x=0.25,
            #     y=0.05,
            #     xanchor="left",
            #     yanchor="bottom",
            #     pad=dict(t=0, b=10),
            # ),
            # dict(
            #     buttons=color_buttons[2],
            #     direction="up",
            #     showactive=True,
            #     x=0.45,
            #     y=0.05,
            #     xanchor="left",
            #     yanchor="bottom",
            #     pad=dict(t=0, b=10),
            # ),
        ],
    )

    return fig
