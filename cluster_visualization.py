import json
import logging
import os
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from umap import UMAP  # GPU versions

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
        df["ids"].values,
        np.concatenate(df["embedding"].values),
        list(*df["error"].values),
    )


def log_plot_plotly(embeddings, labels, label_names, sizes, title, run):
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


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_colormap(n_classes, cmap_name="husl"):
    """Generate colors using Set3, husl, or viridis."""
    if cmap_name == "Set3":
        base_cmap = plt.get_cmap("Set3")
        base_colors = base_cmap.colors
        colors = (base_colors * (n_classes // len(base_colors) + 1))[:n_classes]
    elif cmap_name == "husl":
        colors = sns.color_palette("husl", n_classes)
    elif cmap_name == "viridis":
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / n_classes) for i in range(n_classes)]
    else:
        colors = sns.color_palette("husl", n_classes)

    return [mcolors.rgb2hex(c[:3]) for c in colors]


def add_threshold_input(fig, encoding_name, encoding_info, embeddings, traces):
    """
    Adds a text input box (HTML) + Apply button for thresholding continuous variables.
    """
    # Initial threshold
    init_val = encoding_info["min"]

    # Add HTML input element (Plotly supports foreign HTML in annotations)
    fig.add_annotation(
        dict(
            x=0.01,
            y=1.10,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=(
                f"<b>{encoding_name} threshold:</b> "
                f"<input id='thr_input' type='number' value='{init_val}' "
                f"step='0.01' style='width:80px;'>"
            ),
        )
    )

    # Create button that reads JS from the HTML input
    fig.update_layout(
        updatemenus=fig.layout.updatemenus
        + (
            [
                dict(
                    type="buttons",
                    direction="right",
                    x=0.35,
                    y=1.10,
                    xanchor="left",
                    yanchor="middle",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Apply threshold",
                            method="update",
                            args=[{}, {}],  # We fill via JS hack below
                            execute=True,
                        )
                    ],
                )
            ]
        )
    )

    # JS callback (Plotly hack)
    # This injects JS that replaces x,y,colors based on threshold
    fig.add_annotation(
        dict(
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=(
                "<script>"
                "document.querySelectorAll('g.button').forEach(btn => {"
                "  btn.addEventListener('click', () => {"
                f"    let thr = parseFloat(document.getElementById('thr_input').value);"
                "    let gd = document.querySelector('.js-plotly-plot');"
                "    let d = gd.data;"
                "    for (let i = 0; i < d.length; i++) {"
                f"      if (i >= 0 && i < {len(traces)}) {{"
                f"        let vals = {encoding_info['values'].tolist()};"
                f"        let emb_list = {[emb.tolist() for emb in embeddings]};"
                "        // Find trace-specific mask"
                "      }"
                "    }"
                "    Plotly.react(gd, d, gd.layout);"
                "  });"
                "});"
                "</script>"
            ),
        )
    )


def log_plotly(
    df,
    umap_col=["umap_x", "umap_y"],
    tsne_col=["tsne_x", "tsne_y"],
    class_cols=None,
    superclass_cols=None,
    continuous_cols=None,
    title="Embedding Visualization",
    run=None,
):
    """
    Create interactive embedding plot from DataFrame with one-hot encoded categories.

    Args:
        df: DataFrame with embeddings and labels
        umap_col: List of [x_col, y_col] for UMAP coordinates
        tsne_col: List of [x_col, y_col] for t-SNE coordinates
        class_cols: List of column names for one-hot encoded classes
        superclass_cols: List of column names for one-hot encoded superclasses (optional)
        continuous_cols: List of column names for continuous variables (optional)
        title: Plot title
    """
    # Extract data
    umap_embedding = df[umap_col].values
    tsne_embedding = df[tsne_col].values
    embeddings = [umap_embedding, tsne_embedding]

    # Fixed point size
    norm_sizes = np.full(len(df), 2)

    # Prepare color encodings
    color_options = {}

    # 1. Superclasses (Set3 colormap)
    if superclass_cols is not None:
        # Convert one-hot to categorical
        superclass_matrix = df[superclass_cols].values
        superclasses = np.argmax(superclass_matrix, axis=1)
        superclass_names = superclass_cols
        superclass_colors = get_colormap(len(superclass_names), "Set3")
        color_options["superclasses"] = {
            "values": superclasses,
            "names": superclass_names,
            "colors": superclass_colors,
            "type": "categorical",
        }

    # 2. Classes (husl colormap)
    if class_cols is not None:
        # Convert one-hot to categorical
        class_matrix = df[class_cols].values
        classes = np.argmax(class_matrix, axis=1)
        class_names = class_cols
        class_colors = get_colormap(len(class_names), "husl")
        color_options["classes"] = {
            "values": classes,
            "names": class_names,
            "colors": class_colors,
            "type": "categorical",
        }

    # 3. Continuous variables (viridis colormap)
    if continuous_cols is not None:
        for col in continuous_cols:
            if col in df.columns:
                color_options[col] = {
                    "values": df[col].values,
                    "names": None,
                    "colors": None,
                    "type": "continuous",
                    "colorscale": "Viridis",
                    "min": df[col].min(),
                    "max": df[col].max(),
                }

    # Create figure
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["UMAP", "t-SNE"], horizontal_spacing=0.1
    )

    # Store all traces for each color encoding
    traces_by_encoding = {}

    for encoding_name, encoding_info in color_options.items():
        traces = []

        if encoding_info["type"] == "categorical":
            # Categorical encoding - separate trace per category for toggle functionality
            for idx, (embedding, emb_title) in enumerate(
                zip(embeddings, ["UMAP", "t-SNE"]), 1
            ):
                for cat_idx, cat_name in enumerate(encoding_info["names"]):
                    mask = encoding_info["values"] == cat_idx
                    trace = go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode="markers",
                        marker=dict(
                            size=norm_sizes[mask],
                            color=encoding_info["colors"][cat_idx],
                            opacity=0.7,
                            line=dict(width=0.5, color="white"),
                            sizemode="diameter",
                        ),
                        name=cat_name,
                        legendgroup=cat_name,
                        showlegend=(idx == 1),
                        hovertemplate=f"<b>{cat_name}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>",
                        visible=(encoding_name == "classes"),
                    )
                    traces.append(trace)
                    fig.add_trace(trace, row=1, col=idx)

        else:
            # Continuous encoding
            for idx, (embedding, emb_title) in enumerate(
                zip(embeddings, ["UMAP", "t-SNE"]), 1
            ):
                trace = go.Scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode="markers",
                    marker=dict(
                        size=norm_sizes,
                        color=encoding_info["values"],
                        colorscale=encoding_info["colorscale"],
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                        sizemode="diameter",
                        showscale=(idx == 2),
                        colorbar=dict(title=encoding_name),
                        cmin=encoding_info["min"],
                        cmax=encoding_info["max"],
                    ),
                    name=encoding_name,
                    showlegend=False,
                    hovertemplate=f"<b>{encoding_name}: %{{marker.color:.2f}}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>",
                    visible=(encoding_name == "classes"),
                    customdata=np.column_stack([embedding, encoding_info["values"]]),
                )
                traces.append(trace)
                fig.add_trace(trace, row=1, col=idx)

        traces_by_encoding[encoding_name] = (traces, encoding_info)

    # Create color encoding dropdown buttons
    color_buttons = []

    for encoding_name, (traces, encoding_info) in traces_by_encoding.items():
        visible_list = []
        for trace in fig.data:
            visible_list.append(trace in traces)
        if encoding_info["type"] == "continuous":
            add_threshold_input(fig, encoding_name, encoding_info, embeddings, traces)
        color_buttons.append(
            dict(
                label=encoding_name.replace("_", " ").title(),
                method="update",
                args=[
                    {"visible": visible_list},
                    {
                        "showlegend": encoding_info["type"] == "categorical",
                    },
                ],
            )
        )

    fig.update_layout(
        height=600,
        width=1200,
        hovermode="closest",
        title=title,
        legend=dict(title="Categories (click to toggle)", itemsizing="constant"),
        updatemenus=[
            dict(
                buttons=color_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
    )
    # Point size slider
    size_steps = []
    for size_multiplier in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        step = dict(
            method="restyle",
            args=[{"marker.size": [norm_sizes * size_multiplier]}],
            label=f"{size_multiplier}x",
        )
        size_steps.append(step)

    # Figure size buttons
    size_buttons = []
    for width, height, label in [
        (800, 400, "Small"),
        (1200, 600, "Medium"),
        (1600, 800, "Large"),
        (2000, 1000, "XLarge"),
    ]:
        size_buttons.append(
            dict(
                label=label,
                method="relayout",
                args=[{"width": width, "height": height}],
            )
        )

    fig.update_layout(
        height=FIGURE_HEIGHT,
        width=FIGURE_WIDTH,
        hovermode="closest",
        legend=dict(title="Categories (click to toggle)", itemsizing="constant"),
        updatemenus=[
            # Dropdown for choosing color encoding
            dict(
                buttons=color_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            ),
            # Figure size buttons
            dict(
                buttons=size_buttons,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.50,
                xanchor="center",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                type="buttons",
            ),
        ],
        sliders=[
            dict(
                active=3,  # Default to 1.0x
                currentvalue={"prefix": "Point Size: "},
                pad={"t": 50},
                steps=size_steps,
            )
        ],
    )

    run.log({f"{title} plot": fig})


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
    db_path = "embeddings"
    tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db")
    ]  # extend as needed
    logger = get_logger(args.debug)
    if not args.debug:
        run = wandb.init(
            entity="miro-unet",
            project="VLLM clustering",
            # mode="offline",
            name=f"visu",  # optional descriptive name
        )
    correlation = []
    for table in tables:
        # Start a new wandb run per model
        logger.debug(table)
        ids, X, errors = load_embeddings(table)
        logger.debug(f"{X.shape, X.__class__.__name__}")
        # GPU UMAP
        umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_umap = umap.fit_transform(X)

        # GPU t-SNE
        tsne_gpu = TSNE(
            n_components=2, perplexity=30, learning_rate=200, random_state=42
        )
        embedding_tsne = tsne_gpu.fit_transform(X)
        if run:
            # --- Load COCO JSON ---
            with open(
                "/globalscratch/ucl/irec/darimez/dino/coco/validation/annotations/instances_val2017.json",
                "r",
            ) as f:
                data = json.load(f)

            # --- Build lookup maps ---
            id_to_name = {
                cat["id"]: cat["supercategory"] + "_" + cat["name"]
                for cat in data["categories"]
            }
            id_to_super = {
                cat["id"]: cat["supercategory"] for cat in data["categories"]
            }

            # Define category and supercategory lists
            categories = list(set(id_to_name.values()))
            supercategories = list(set(id_to_super.values()))

            # --- Group annotations by image ---
            anns_by_image = defaultdict(list)
            for ann in data["annotations"]:
                anns_by_image[ann["image_id"]].append(ann)

            rows = []
            for emb_umap, emb_tsne, error, img_id in zip(
                embedding_umap, embedding_tsne, errors, ids
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
                    + list(error)
                )
                rows.append(row)

            # Build dataframe
            columns = (
                [f"UMAP-{i}" for i in range(embedding_umap.shape[1])]
                + [f"t-SNE-{i}" for i in range(embedding_tsne.shape[1])]
                + categories
                + supercategories
                + ["box_loss", "cls_loss", "dfl_loss", "super_cls_loss"]
            )
            df = pd.DataFrame(rows, columns=columns)

            # Select only UMAP and t-SNE columns
            umap_cols = [f"UMAP-{i}" for i in range(embedding_umap.shape[1])]
            tsne_cols = [f"t-SNE-{i}" for i in range(embedding_tsne.shape[1])]

            # Or just cross-correlations (UMAP vs t-SNE only)
            cross_corr = df[umap_cols].corrwith(df[tsne_cols])
            correlation.append(cross_corr)

            # Log to W&B
            run.log({f"{table}": wandb.Table(dataframe=df)})
            log_plot_plotly(
                [embedding_umap, embedding_tsne],
                df[supercategories].values.argmax(1),
                supercategories,
                df["super_cls_loss"],
                f"{table}",
                run,
            )

    if run:
        correlation_df = pd.concat(correlation, axis=1)
        run.log({"correlation": wandb.Table(dataframe=correlation_df)})
        run.finish()
