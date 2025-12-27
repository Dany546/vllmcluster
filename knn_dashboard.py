"""
Interactive Dash dashboard for KNN clustering results filtering.
Allows users to toggle traces by text query (substring match, case-insensitive).

Usage:
    # Run on cluster (default):
    python knn_dashboard.py
    
    # Run locally with local DB copy:
    python knn_dashboard.py --local --db-path ~/local_knn.db
    
    # Run with custom DB path:
    python knn_dashboard.py --db-path /path/to/knn_results.db
"""

import argparse
import sqlite3
import json
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive KNN Results Dashboard"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to KNN results database (SQLite). "
             "Default: /globalscratch/ucl/irec/darimez/dino/knn_results.db (cluster)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local database copy (expects ~/knn_results.db)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run Dash server on. Default: 127.0.0.1 (local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for Dash server. Default: 8050",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    return parser.parse_args()


def get_db_path(args):
    """Determine database path based on arguments."""
    if args.db_path:
        # User specified explicit path
        return args.db_path
    elif args.local:
        # Use local copy
        local_path = os.path.expanduser("~/knn_results.db")
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Local database not found at {local_path}. "
                "Copy it from cluster first: "
                "rsync -avz user@cluster:/globalscratch/.../knn_results.db ~/"
            )
        return local_path
    else:
        # Default: cluster path
        cluster_path = "/globalscratch/ucl/irec/darimez/dino/knn_results.db"
        return cluster_path


# Parse command-line arguments early
args = parse_args()
DB_PATH = get_db_path(args)

print(f"[KNN Dashboard] Using database: {DB_PATH}")
print(f"[KNN Dashboard] Serving on http://{args.host}:{args.port}")



def load_knn_results():
    """Load KNN results from SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM knn_results ORDER BY model", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading KNN results: {e}")
        return pd.DataFrame()


def create_initial_figure(df, x_axis="k"):
    """Create the initial Plotly figure with KNN results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        KNN results dataframe
    x_axis : str
        Column name to display on x-axis. Default: "k"
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No KNN results found in database.")
        return fig

    # Make a copy to avoid modifying the original
    df = df.copy()
    
    print(f"[DEBUG] Columns in dataframe: {df.columns.tolist()}")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    
    # Merge model and distance_metric for cleaner trace names (if distance_metric exists)
    if "distance_metric" in df.columns:
        df["model_full"] = df[["model", "distance_metric"]].astype(str).agg("_".join, axis=1)
    else:
        df["model_full"] = df["model"]

    grid = {
        "target": ["hit_freq", "mean_iou", "mean_conf", "flag_cat", "flag_supercat"],
        "target_name": [
            "Hit Frequency",
            "Mean IoU",
            "Mean Confidence",
            "Categories",
            "Super Categories",
        ],
        "metrics": ["corr", "mae", "r2"],
        "metrics_name": ["Correlation", "Mean Absolute Error", "R-squared"],
    }

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
        "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52BE80"
    ]

    fig = go.Figure()

    models = [m for m in df["model_full"].unique() if m not in ("clip", "dino")]
    n_models = len(models)
    n_targets = len(grid["target"])
    n_metrics = len(grid["metrics"])

    print(f"[DEBUG] Found {n_models} models: {models}")

    # Add traces: metric × target × model
    for metric, target in [(m, t) for m in grid["metrics"] for t in grid["target"]]:
        for i, model in enumerate(models):
            sub = df[(df["model_full"] == model) & (df["target"] == target)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Box(
                    x=sub[x_axis],
                    y=sub[metric],
                    name=f"{model}_{target}",  # trace name for filtering
                    marker_color=colors[i % len(colors)],
                    boxmean=True,
                    visible=(metric == "corr" and target == "mean_iou"),
                )
            )

    metric_ranges = {
        "corr": [-1, 1],
        "mae": [0, df["mae"].max() + 0.05 if "mae" in df.columns else 1],
        "r2": [df["r2"].min() - 0.05 if "r2" in df.columns else 0, 1],
    }

    fig.update_layout(
        title="KNN Results - Correlation (Mean IoU)",
        xaxis_title=x_axis,
        yaxis_title="Correlation",
        yaxis_range=metric_ranges["corr"],
        template="plotly_white",
        boxmode="group",
        height=700,
        hovermode="closest",
    )

    return fig


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data
df = load_knn_results()
initial_figure = create_initial_figure(df, x_axis="k")

# Store trace names for efficient filtering
trace_names = [tr.name for tr in initial_figure.data]

# Available k values to filter
k_values = sorted(df["k"].unique().tolist()) if not df.empty else []
k_options = [{"label": str(k), "value": k} for k in k_values]

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("KNN Results Explorer", className="mb-3"),
                        html.P(
                            "Select k values to display, then filter traces by name substring (e.g., 'dino', 'yolo').",
                            className="text-muted",
                        ),
                    ]
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select k values to display:", className="fw-bold"),
                        dcc.Dropdown(
                            id="k-values-selector",
                            options=k_options,
                            value=k_values,
                            multi=True,
                            clearable=True,
                        ),
                    ],
                    md=4,
                    className="mb-3",
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Filter traces by text (case-insensitive):", 
                                   className="fw-bold"),
                        dbc.InputGroup(
                            [
                                dbc.Input(
                                    id="filter-input",
                                    placeholder="e.g., 'dino' or 'mean_iou'",
                                    type="text",
                                    className="form-control",
                                ),
                                dbc.Button(
                                    "Toggle",
                                    id="toggle-btn",
                                    color="primary",
                                    className="ms-2",
                                ),
                                dbc.Button(
                                    "Show All",
                                    id="show-all-btn",
                                    color="success",
                                    className="ms-2",
                                ),
                                dbc.Button(
                                    "Hide All",
                                    id="hide-all-btn",
                                    color="danger",
                                    className="ms-2",
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                    md=12,
                )
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="knn-figure"),
                    ],
                    md=12,
                )
            ],
        ),
        # Hidden stores for state
        dcc.Store(id="figure-store", data=initial_figure.to_json()),
        dcc.Store(id="trace-names-store", data=trace_names),
        dcc.Store(id="visibility-store", data=[True] * len(trace_names)),
        dcc.Store(id="df-store", data=df.to_json(orient="split")),
    ],
    fluid=True,
    className="mt-4",
)


@callback(
    Output("knn-figure", "figure"),
    Output("visibility-store", "data"),
    Input("k-values-selector", "value"),
    Input("toggle-btn", "n_clicks"),
    Input("show-all-btn", "n_clicks"),
    Input("hide-all-btn", "n_clicks"),
    State("filter-input", "value"),
    State("df-store", "data"),
    prevent_initial_call=False,
)
def update_figure(k_values, toggle_clicks, show_clicks, hide_clicks, query, df_json):
    """
    Update figure based on selected k values and trace text filtering.
    - k_values: Filter dataframe by selected k values
    - Query: Show/hide traces matching text substring
    """
    if not df_json:
        return initial_figure, []

    # Reconstruct dataframe from store
    df_filtered = pd.read_json(df_json, orient="split")
    
    # Filter by k values if selected
    if k_values:
        df_filtered = df_filtered[df_filtered["k"].isin(k_values)]
    
    # Recreate figure with filtered data
    fig = create_initial_figure(df_filtered, x_axis="k")
    trace_names = [tr.name for tr in fig.data]
    
    # Initialize visibility (all visible by default)
    visibility = [True] * len(trace_names)
    
    # Determine which action was triggered
    ctx_id = None
    if toggle_clicks and toggle_clicks > 0:
        ctx_id = "toggle"
    elif show_clicks and show_clicks > 0:
        ctx_id = "show_all"
    elif hide_clicks and hide_clicks > 0:
        ctx_id = "hide_all"

    if ctx_id == "toggle" and query:
        # Case-insensitive substring match
        q = query.lower()
        visibility = [
            (q in name.lower()) for name in trace_names
        ]
    elif ctx_id == "show_all":
        visibility = [True] * len(trace_names)
    elif ctx_id == "hide_all":
        visibility = [False] * len(trace_names)

    # Update figure with visibility
    for i, vis in enumerate(visibility):
        if i < len(fig.data):
            fig.data[i].visible = vis

    return fig, visibility


@callback(
    Output("filter-input", "value"),
    Input("filter-input", "value"),
    prevent_initial_call=True,
)
def filter_input_handler(value):
    """Handle filter input changes."""
    return value


if __name__ == "__main__":
    app.run(debug=args.debug, host=args.host, port=args.port)
