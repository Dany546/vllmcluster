import json
import logging
import os
import sqlite3
from collections import defaultdict

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from umap import UMAP  # GPU versions
from utils import dict_to_filename, get_lookups, table_exists

import wandb


# Load embeddings from SQL
def load_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    df = pd.read_sql_query("SELECT * FROM embeddings", conn)
    df["embedding"] = df["embedding"].apply(
        lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
    )
    # df["error"] = df["error"].apply(
    #     lambda b: np.frombuffer(b, dtype=np.float32).reshape(1, -1)
    # )
    conn.close()
    return (
        df["img_id"].values,
        np.concatenate(df["embedding"].values),
        df["hit_freq"].values,
        df["mean_iou"].values,
        df["mean_conf"].values,
        df["flag_cat"].values,
        df["flag_supercat"].values,
    )


def log_plot(embeddings, labels, label_names, sizes, title, run):
    # Keep reference to scatter for colorbar
    scatter = None
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    for embedding, ax, ax_title in zip(embeddings, axes, ["UMAP", "t-SNE"]):
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab20",
            s=20,
        )
        ax.set_title(ax_title)
    # Adjust layout to make room for colorbar
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cbar = fig.colorbar(
        scatter,
        ax=axes,
        orientation="vertical",
        ticks=np.arange(len(label_names)),
    )
    cbar.ax.set_yticklabels(label_names)
    run.log({f"{title} plot": wandb.Image(fig)})
    plt.close()


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
    ACTIVE_POINT_SIZE = 10  # Size when class is toggled ON
    INACTIVE_POINT_SIZE = 4  # Size when class is toggled OFF
    BACKGROUND_POINT_SIZE = 4  # Size for permanent background

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


def get_logger(debug):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def visualize_clusters(args):
    # -----------------------------
    # Loop over multiple models/tables
    # -----------------------------
    db_path = f"/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db")
    ]  # extend as needed
    logger = get_logger(args.debug)
    if not args.debug:
        if wandb.run is None:
            # No run yet → initialize a new one
            wandb.init(
                entity="miro-unet",
                project="VLLM clustering",
                # mode="offline",
                name=f"visu",  # optional descriptive name
            )
        run = wandb.run
    correlation = []
    for table in tables:
        logger.debug(table)
        ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)

        hyperparams = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
            },
            "tsne": {
                "n_components": 2,
                "perplexity": 30,
                "early_exaggeration": 12,
                "learning_rate": 200,
                "n_iter": 1000,
            },
        }
        sql_names = [
            dict_to_filename(hyperparams["umap"]),
            dict_to_filename(hyperparams["tsne"]),
        ]
        emb_conn = sqlite3.connect(table.replace("embeddings", "proj"), timeout=30)
        emb_conn.execute("PRAGMA journal_mode=WAL")
        emb_conn.execute("PRAGMA synchronous=OFF")
        emb_conn.execute("PRAGMA temp_store=MEMORY")
        emb_conn.execute("PRAGMA cache_size=-20000")
        emb_cursor = emb_conn.cursor()
        if not table_exists(emb_cursor, f"umap_{sql_names[0]}"):
            sql_columns = ", ".join(
                [
                    "comp_" + str(i) + " REAL"
                    for i in range(hyperparams["umap"]["n_components"])
                ]
            )
            emb_cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS umap_{sql_names[0]} (
                    id INTEGER PRIMARY KEY
                    {sql_columns}
                )""")
            umap = UMAP(**hyperparams["umap"])
            embedding_umap = umap.fit_transform(X)
        else:
            embedding_umap = np.zeros((len(X), 2))
        if not table_exists(emb_cursor, f"tsne_{sql_names[1]}"):
            sql_columns = ", ".join(
                [
                    "comp_" + str(i) + " REAL"
                    for i in range(hyperparams["tsne"]["n_components"])
                ]
            )
            emb_cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS tsne_{sql_names[1]} (
                    id INTEGER PRIMARY KEY
                    {sql_columns}
                )""")
            tsne_gpu = TSNE(**hyperparams["tsne"])
            embedding_tsne = tsne_gpu.fit_transform(X)
        else:
            embedding_tsne = np.zeros((len(X), 2))
        if run:
            # --- Load COCO JSON ---
            anns_by_image, id_to_name, id_to_super, categories, supercategories = (
                get_lookups()
            )
            rows = []
            umap_rows = [] if embedding_umap is not None else None
            tsne_rows = [] if embedding_tsne is not None else None
            for emb_umap, emb_tsne, hf, miou, mconf, cat, supercat, img_id in zip(
                embedding_umap, embedding_tsne, hfs, mious, mconfs, cats, supercats, ids
            ):
                anns = anns_by_image[img_id]  # from COCO JSON grouping

                # Initialize counts
                cat_counts = {name: 0 for name in categories}
                super_counts = {sc: 0 for sc in supercategories}

                # Count instances
                for ann in anns:
                    cat_name = id_to_name[ann["category_id"]]
                    supercat = id_to_super[ann["category_id"]]
                    cat_counts[cat_name] += 1
                    super_counts[supercat] += 1

                # Build row
                row = (
                    list(emb_umap)
                    + list(emb_tsne)
                    + list(cat_counts.values())
                    + list(super_counts.values())
                    + list([hf, miou, mconf])
                )
                rows.append(row)
                if umap_rows:
                    umap_rows.append([len(row)] + list(emb_umap))
                if tsne_rows:
                    tsne_rows.append([len(row)] + list(emb_tsne))
                if len(row) % 500 == 0:
                    if umap_rows:
                        emb_cursor.execute(
                            f"INSERT INTO umap_{sql_names[0]} VALUES ({','.join(['?' for _ in range(len(umap_rows[0]))])})",
                            umap_rows,
                        )
                        umap_rows = [] if embedding_umap is not None else None
                    if tsne_rows:
                        emb_cursor.execute(
                            f"INSERT INTO tsne_{sql_names[1]} VALUES ({','.join(['?' for _ in range(len(tsne_rows[0]))])})",
                            tsne_rows,
                        )
                        tsne_rows = [] if embedding_tsne is not None else None

                    emb_conn.commit()
            if umap_rows:
                emb_cursor.execute(
                    f"INSERT INTO umap_{sql_names[0]} VALUES ({','.join(['?' for _ in range(len(umap_rows[0]))])})",
                    umap_rows,
                )
            if tsne_rows:
                emb_cursor.execute(
                    f"INSERT INTO tsne_{sql_names[1]} VALUES ({','.join(['?' for _ in range(len(tsne_rows[0]))])})",
                    tsne_rows,
                )
            emb_conn.commit()
            # Build dataframe
            columns = (
                [f"UMAP-{i}" for i in range(embedding_umap.shape[1])]
                + [f"t-SNE-{i}" for i in range(embedding_tsne.shape[1])]
                + categories
                + supercategories
                + ["hit_freq", "mean_iou", "mean_conf"]
            )
            df = pd.DataFrame(rows, columns=columns)

            # Select only UMAP and t-SNE columns
            umap_cols = [f"UMAP-{i}" for i in range(embedding_umap.shape[1])]
            tsne_cols = [f"t-SNE-{i}" for i in range(embedding_tsne.shape[1])]

            # Or just cross-correlations (UMAP vs t-SNE only)
            cross_corr = df[umap_cols].corrwith(df[tsne_cols])
            correlation.append(cross_corr)

            # Log to W&B
            table_name = table.split(f"{os.sep}")[-1].split(".")[0]
            run.log({f"{table_name}": wandb.Table(dataframe=df)})
            log_plot(
                [embedding_umap, embedding_tsne],
                df[supercategories].values.argmax(1),
                supercategories,
                df["hit_freq"],
                f"{table_name}",
                run,
            )
            fig = log_plot_plotly(
                df,
                umap_col=["UMAP-0", "UMAP-1"],
                tsne_col=["t-SNE-0", "t-SNE-1"],
                class_cols=categories,
                superclass_cols=supercategories,
                continuous_cols=["hit_freq", "mean_iou", "mean_conf"],
                title="Embedding Visualization",
            )
            run.log({f"{table_name}": fig})

    if run:
        correlation_df = pd.concat(correlation, axis=1)
        run.log({"correlation": wandb.Table(dataframe=correlation_df)})
        run.finish()
