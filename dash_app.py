import os
import sqlite3

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots
from utils import get_lookups

# ============================================
# CONFIGURATION - Customize appearance here
# ============================================
ACTIVE_POINT_SIZE = 6  # Size when category is toggled ON
INACTIVE_POINT_SIZE = 4  # Size when category is toggled OFF
BACKGROUND_POINT_SIZE = 3  # Size for permanent background layer

ACTIVE_OPACITY = 0.9  # Opacity when toggled ON
INACTIVE_OPACITY = 0.3  # Opacity when toggled OFF
BACKGROUND_OPACITY = 0.25  # Opacity for permanent background

INACTIVE_COLOR = "grey"  # Color when toggled OFF
BACKGROUND_COLOR = "grey"  # Color for permanent background

DB_PATH = "../data"


# ============================================
# DB HELPERS
# ============================================
def get_runs_metadata():
    runs = []
    for algo in ["umap", "tsne"]:
        conn = sqlite3.connect(os.path.join(DB_PATH, f"{algo}.db"))
        df = pd.read_sql("SELECT * FROM metadata", conn)
        conn.close()
        df["algo"] = algo
        runs.append(df)
    return pd.concat(runs)


def fetch_embeddings(run_id, algo):
    conn = sqlite3.connect(os.path.join(DB_PATH, f"{algo}.db"))
    df = pd.read_sql(
        "SELECT run_id, img_id, x, y FROM embeddings WHERE run_id=?",
        conn,
        params=(run_id,),
    )
    conn.close()
    return df


def fetch_metrics(model_name):
    conn = sqlite3.connect(os.path.join(DB_PATH, "metrics.db"))
    df = pd.read_sql("SELECT * FROM metrics WHERE model=?", conn, params=(model_name,))
    conn.close()
    return df


# ============================================
# DASH APP INITIALIZATION
# ============================================
app = dash_app.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        # ============================================
        # HEADER SECTION
        # ============================================
        html.Div(
            [
                html.H2("Advanced Embedding Explorer", style={"margin-bottom": "10px"}),
                html.P(
                    "Interactive multi-plot embedding visualization with cross-filtering and dynamic controls",
                    style={"color": "#666", "font-style": "italic"},
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        # ============================================
        # GLOBAL CONTROLS
        # ============================================
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Run:", style={"font-weight": "bold"}),
                        dcc.Dropdown(id="run-choice", style={"width": "100%"}),
                    ],
                    style={
                        "width": "60%",
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Add Plot",
                            id="add-plot-btn",
                            n_clicks=0,
                            style={
                                "padding": "10px 20px",
                                "background": "#4CAF50",
                                "color": "white",
                                "border": "none",
                                "border-radius": "4px",
                                "cursor": "pointer",
                                "font-size": "14px",
                            },
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "vertical-align": "top",
                        "margin-top": "25px",
                    },
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        # ============================================
        # GLOBAL FILTER CONTROLS (for continuous variables)
        # ============================================
        html.Div(
            id="global-filters",
            style={
                "padding": "20px",
                "background": "#f9f9f9",
                "border-radius": "8px",
                "margin-bottom": "20px",
                "display": "none",  # Hidden until data loaded
            },
        ),
        # ============================================
        # PLOT CONTAINER (dynamically populated)
        # ============================================
        html.Div(id="plots-container", children=[]),
        # ============================================
        # HIDDEN STORES for state management
        # ============================================
        dcc.Store(id="selected-points-store", data={}),  # Store cross-filter selections
        dcc.Store(id="plot-counter", data=0),  # Track number of plots
        dcc.Store(id="data-store", data=None),  # Store merged dataframe
    ],
    style={"padding": "20px", "max-width": "1600px", "margin": "0 auto"},
)


# ============================================
# CALLBACK: Populate run dropdown
# ============================================
@app.callback(Output("run-choice", "options"), Input("run-choice", "id"))
def populate_dropdown(_):
    """Populate run selection dropdown with available runs"""
    meta = get_runs_metadata()
    options = []
    for _, row in meta.iterrows():
        hp_keys = [c for c in row.index if c not in ["run_id", "model", "algo"]]
        hp_str = ", ".join([f"{k}={row[k]}" for k in hp_keys if pd.notnull(row[k])])
        label = f"{row.model} | {row.algo} | {hp_str}"
        options.append({"label": label, "value": row.run_id})
    return options


# ============================================
# CALLBACK: Load data and create global filters
# ============================================
@app.callback(
    [
        Output("data-store", "data"),
        Output("global-filters", "children"),
        Output("global-filters", "style"),
    ],
    Input("run-choice", "value"),
)
def load_data_and_filters(run_id):
    """
    Load embedding and metrics data, create global filter controls for continuous variables
    """
    if not run_id:
        return None, [], {"display": "none"}

    # Fetch data
    meta = get_runs_metadata()
    row = meta.loc[meta.run_id == run_id].iloc[0]
    algo = row.algo
    model = row.model

    emb = fetch_embeddings(run_id, algo)
    metrics = fetch_metrics(model)
    df = pd.merge(emb, metrics, on="img_id", how="left")

    # Identify continuous and categorical columns
    exclude_cols = ["id", "run_id", "img_id", "model", "x", "y"]
    available_cols = [c for c in df.columns if c not in exclude_cols]

    continuous_cols = []
    categorical_cols = []

    for col in available_cols:
        if df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 10:
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    # Create global filter controls for continuous variables
    filter_controls = [html.H4("Global Filters", style={"margin-bottom": "15px"})]

    for col in continuous_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())

        filter_controls.append(
            html.Div(
                [
                    html.Label(
                        f"{col}:", style={"font-weight": "bold", "margin-bottom": "5px"}
                    ),
                    dcc.RangeSlider(
                        id={"type": "global-filter", "col": col},
                        min=col_min,
                        max=col_max,
                        value=[col_min, col_max],
                        marks={col_min: f"{col_min:.2f}", col_max: f"{col_max:.2f}"},
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=(col_max - col_min) / 100,
                    ),
                ],
                style={"margin-bottom": "20px"},
            )
        )

    if not continuous_cols:
        filter_controls.append(
            html.P(
                "No continuous variables available for filtering",
                style={"color": "#999", "font-style": "italic"},
            )
        )

    return (
        df.to_dict("records"),
        filter_controls,
        {
            "padding": "20px",
            "background": "#f9f9f9",
            "border-radius": "8px",
            "margin-bottom": "20px",
            "display": "block",
        },
    )


# ============================================
# CALLBACK: Add new plot
# ============================================
@app.callback(
    [Output("plots-container", "children"), Output("plot-counter", "data")],
    [
        Input("add-plot-btn", "n_clicks"),
        Input({"type": "remove-plot", "index": ALL}, "n_clicks"),
    ],
    [
        State("plots-container", "children"),
        State("plot-counter", "data"),
        State("data-store", "data"),
    ],
)
def manage_plots(add_clicks, remove_clicks, current_plots, counter, data):
    """
    Dynamically add or remove plots from the interface
    """
    if not data:
        return [], 0

    df = pd.DataFrame(data)
    exclude_cols = ["id", "run_id", "img_id", "model"]
    available_cols = [c for c in df.columns if c not in exclude_cols]

    # Determine which button was clicked
    triggered_id = ctx.triggered_id

    if triggered_id == "add-plot-btn":
        # Add a new plot
        plot_id = counter

        new_plot = html.Div(
            [
                # Plot controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "X-axis:",
                                    style={
                                        "margin-right": "10px",
                                        "font-weight": "bold",
                                    },
                                ),
                                dcc.Dropdown(
                                    id={"type": "x-axis", "index": plot_id},
                                    options=[
                                        {"label": c, "value": c} for c in available_cols
                                    ],
                                    value="x",
                                    style={"width": "200px"},
                                ),
                            ],
                            style={"display": "inline-block", "margin-right": "20px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Y-axis:",
                                    style={
                                        "margin-right": "10px",
                                        "font-weight": "bold",
                                    },
                                ),
                                dcc.Dropdown(
                                    id={"type": "y-axis", "index": plot_id},
                                    options=[
                                        {"label": c, "value": c} for c in available_cols
                                    ],
                                    value="y",
                                    style={"width": "200px"},
                                ),
                            ],
                            style={"display": "inline-block", "margin-right": "20px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Color:",
                                    style={
                                        "margin-right": "10px",
                                        "font-weight": "bold",
                                    },
                                ),
                                dcc.Dropdown(
                                    id={"type": "color-axis", "index": plot_id},
                                    options=[{"label": "None", "value": "none"}]
                                    + [
                                        {"label": c, "value": c} for c in available_cols
                                    ],
                                    value="none",
                                    style={"width": "200px"},
                                ),
                            ],
                            style={"display": "inline-block", "margin-right": "20px"},
                        ),
                        html.Button(
                            "Remove",
                            id={"type": "remove-plot", "index": plot_id},
                            style={
                                "padding": "8px 16px",
                                "background": "#f44336",
                                "color": "white",
                                "border": "none",
                                "border-radius": "4px",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                    style={
                        "margin-bottom": "15px",
                        "padding": "10px",
                        "background": "#fff",
                        "border-radius": "4px",
                    },
                ),
                # Plot graph
                dcc.Graph(
                    id={"type": "plot-graph", "index": plot_id},
                    style={"height": "500px"},
                    config={"displayModeBar": True},
                ),
            ],
            style={
                "margin-bottom": "30px",
                "padding": "20px",
                "border": "1px solid #ddd",
                "border-radius": "8px",
                "background": "white",
            },
            id={"type": "plot-container", "index": plot_id},
        )

        if current_plots is None:
            current_plots = []
        current_plots.append(new_plot)
        return current_plots, counter + 1

    elif triggered_id and triggered_id.get("type") == "remove-plot":
        # Remove the clicked plot
        remove_index = triggered_id["index"]
        updated_plots = [
            p for p in current_plots if p["props"]["id"]["index"] != remove_index
        ]
        return updated_plots, counter

    return current_plots or [], counter


# ============================================
# CALLBACK: Update all plots (main visualization logic)
# ============================================
@app.callback(
    Output({"type": "plot-graph", "index": ALL}, "figure"),
    [
        Input({"type": "x-axis", "index": ALL}, "value"),
        Input({"type": "y-axis", "index": ALL}, "value"),
        Input({"type": "color-axis", "index": ALL}, "value"),
        Input({"type": "global-filter", "col": ALL}, "value"),
        Input({"type": "plot-graph", "index": ALL}, "selectedData"),
    ],
    [State("data-store", "data"), State("run-choice", "value")],
)
def update_all_plots(
    x_axes, y_axes, color_axes, filter_ranges, selected_data, data, run_id
):
    """
    Update all plots based on:
    - Selected axes
    - Color variables
    - Global filters (range sliders)
    - Cross-filtering (selections from other plots)
    """
    if not data or not run_id:
        return [go.Figure()] * len(x_axes)

    df = pd.DataFrame(data)

    # Get metadata for title
    meta = get_runs_metadata()
    row = meta.loc[meta.run_id == run_id].iloc[0]
    algo = row.algo
    model = row.model

    # ============================================
    # APPLY GLOBAL FILTERS (from range sliders)
    # ============================================
    triggered = ctx.triggered_id
    filter_cols = [t["col"] for t in ctx.inputs_list[3]] if ctx.inputs_list[3] else []

    for col, (min_val, max_val) in zip(filter_cols, filter_ranges):
        if col in df.columns:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]

    # ============================================
    # APPLY CROSS-FILTERING (from plot selections)
    # ============================================
    # If any plot has selected points, filter to only those
    all_selected_indices = set()
    for sel_data in selected_data:
        if sel_data and "points" in sel_data:
            indices = [p["pointIndex"] for p in sel_data["points"]]
            if all_selected_indices:
                all_selected_indices &= set(indices)  # Intersection
            else:
                all_selected_indices = set(indices)

    if all_selected_indices:
        df = df.iloc[list(all_selected_indices)]

    # ============================================
    # CREATE FIGURES for each plot
    # ============================================
    figures = []

    for idx, (x_col, y_col, color_col) in enumerate(zip(x_axes, y_axes, color_axes)):
        if x_col not in df.columns or y_col not in df.columns:
            figures.append(go.Figure())
            continue

        fig = go.Figure()

        # Background layer - always visible
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    size=BACKGROUND_POINT_SIZE,
                    color=BACKGROUND_COLOR,
                    opacity=BACKGROUND_OPACITY,
                    line=dict(width=0),
                ),
                name="Background",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # ============================================
        # NO COLOR selected
        # ============================================
        if color_col == "none" or color_col not in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    marker=dict(
                        size=ACTIVE_POINT_SIZE,
                        color="steelblue",
                        opacity=ACTIVE_OPACITY,
                        line=dict(width=0.5, color="white"),
                    ),
                    showlegend=False,
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                )
            )

        # ============================================
        # CONTINUOUS COLOR variable
        # ============================================
        elif (
            df[color_col].dtype in [np.float64, np.int64]
            and df[color_col].nunique() > 10
        ):
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    marker=dict(
                        size=ACTIVE_POINT_SIZE,
                        color=df[color_col],
                        colorscale="Viridis",
                        opacity=ACTIVE_OPACITY,
                        line=dict(width=0.5, color="white"),
                        showscale=True,
                        colorbar=dict(title=color_col, len=0.7),
                    ),
                    showlegend=False,
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{color_col}: %{{marker.color}}<extra></extra>",
                )
            )

        # ============================================
        # CATEGORICAL COLOR variable (dual-trace system)
        # ============================================
        else:
            unique_vals = sorted(df[color_col].dropna().unique())
            colors = px.colors.qualitative.Set2

            for cat_idx, val in enumerate(unique_vals):
                mask = df[color_col] == val
                color = colors[cat_idx % len(colors)]

                # ACTIVE trace (colored)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask, x_col],
                        y=df.loc[mask, y_col],
                        mode="markers",
                        marker=dict(
                            size=ACTIVE_POINT_SIZE,
                            color=color,
                            opacity=ACTIVE_OPACITY,
                            line=dict(width=0.5, color="white"),
                        ),
                        name=str(val),
                        legendgroup=str(val),
                        showlegend=True,
                        hovertemplate=f"{color_col}: {val}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                        visible=True,
                    )
                )

                # INACTIVE trace (grey)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask, x_col],
                        y=df.loc[mask, y_col],
                        mode="markers",
                        marker=dict(
                            size=INACTIVE_POINT_SIZE,
                            color=INACTIVE_COLOR,
                            opacity=INACTIVE_OPACITY,
                            line=dict(width=0),
                        ),
                        name=str(val),
                        legendgroup=str(val),
                        showlegend=False,
                        hovertemplate=f"{color_col}: {val}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                        visible="legendonly",
                    )
                )

        # Layout configuration
        legend_title = ""
        if color_col != "none" and color_col in df.columns:
            if not (
                df[color_col].dtype in [np.float64, np.int64]
                and df[color_col].nunique() > 10
            ):
                legend_title = f"{color_col}<br><span style='font-size:10px'>Click to toggle</span>"

        fig.update_layout(
            title=f"Plot {idx + 1}: {model} ({algo.upper()})",
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode="closest",
            dragmode="select",  # Enable box/lasso selection
            legend=dict(title=legend_title, itemsizing="constant"),
            margin=dict(l=50, r=50, t=50, b=50),
        )

        figures.append(fig)

    return figures


if __name__ == "__main__":
    app.run_server(debug=True)
