import os
import sqlite3

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from utils import get_lookups

# Adjust this to the folder where your .db files live
DB_PATH = "/data"


# --- DB helpers ---
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


# --- Dash layout ---
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H3("Naive Embedding Explorer"),
        dcc.Dropdown(id="run-choice", style={"width": "70%"}),
        dcc.Dropdown(
            id="color-choice",
            style={"width": "40%"},
            placeholder="Select metric/column for coloring",
        ),
        dcc.Graph(id="embedding-plot", style={"height": "800px"}),
    ]
)


@app.callback(Output("run-choice", "options"), Input("run-choice", "id"))
def populate_dropdown(_):
    meta = get_runs_metadata()
    options = []
    for _, row in meta.iterrows():
        # Collect hyperparams from metadata row (excluding run_id, model, algo)
        hp_keys = [c for c in row.index if c not in ["run_id", "model", "algo"]]
        hp_str = ", ".join([f"{k}={row[k]}" for k in hp_keys if pd.notnull(row[k])])
        label = f"{row.model} | {row.algo} | {hp_str}"
        # Value is just the run_id (we’ll use it later)
        options.append({"label": label, "value": row.run_id})
    return options


# --- Populate color dropdown based on metrics columns ---
@app.callback(Output("color-choice", "options"), Input("run-choice", "value"))
def populate_color_options(choice):
    if not choice:
        return []
    run_id, _ = choice
    metrics = fetch_metrics(run_id)
    # Exclude id/run_id/img_id/model
    cols = [c for c in metrics.columns if c not in ["id", "run_id", "img_id", "model"]]
    return [{"label": c, "value": c} for c in cols]


# --- Update plot ---
@app.callback(
    Output("embedding-plot", "figure"),
    [Input("run-choice", "value"), Input("color-choice", "value")],
)
def update_plot(run_id, color_col):
    if not run_id:
        return go.Figure()

    # Look up algo + model from metadata
    meta = get_runs_metadata()
    row = meta.loc[meta.run_id == run_id].iloc[0]
    algo = row.algo
    model = row.model

    emb = fetch_embeddings(run_id, algo)
    metrics = fetch_metrics(model)

    categories, supercategories = get_lookups()[-2:]

    # Align by img_id
    df = pd.merge(emb, metrics, on="img_id", how="left")

    if color_col and color_col in df.columns:
        # If the chosen column is continuous → one scatter with colorscale
        if df[color_col].dtype in [np.float64, np.int64]:
            fig = px.scatter(df, x="x", y="y", color=color_col, hover_data=[color_col])
            fig.update_layout(
                legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
                title=f"{algo.upper()} Embedding for {model}",
            )
            return fig
        # If the chosen column is categorical (e.g. counts) → multiple traces
        else:
            fig = go.Figure()
            unique_vals = df[color_col].unique()
            colors = px.colors.qualitative.Set2
            for idx, val in enumerate(unique_vals):
                mask = df[color_col] == val
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask, "x"],
                        y=df.loc[mask, "y"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=colors[idx % len(colors)],
                            line=dict(width=0.5, color="white"),
                        ),
                        name=str(val),
                        legendgroup=str(val),
                        hovertext=categories[val]
                        if color_col in categories
                        else supercategories[val],
                        hoverinfo="text",
                    )
                )
            fig.update_layout(
                legend=dict(
                    title=color_col, itemclick="toggle", itemdoubleclick="toggleothers"
                ),
                title=f"{algo.upper()} Embedding for {model} ",
            )
            return fig

    # Default scatter if no color chosen
    fig = px.scatter(df, x="x", y="y", hover_data=["img_id"])
    fig.update_layout(title=f"{algo.upper()} Embedding for {model} ")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
