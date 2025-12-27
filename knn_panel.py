"""
Prototype Panel/HoloViz app for KNN results grid editor.
Run with: `panel serve knn_panel.py --show`
"""
import os
import sqlite3
import pandas as pd
import hvplot.pandas  # noqa: F401
import panel as pn
import holoviews as hv
from holoviews import opts

pn.extension('plotly')

DB_PATH = "/globalscratch/ucl/irec/darimez/dino/knn_results.db"


def load_df(path=DB_PATH):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT * FROM knn_results ORDER BY model", conn)
    conn.close()
    return df


df = load_df()

models = sorted(df['model'].unique())
ks = sorted(df['k'].unique())
metrics = ['corr','mae','r2']
targets = ['hit_freq','mean_iou','mean_conf','flag_cat','flag_supercat']

# Simple grid: pick model for each cell via selectors
rows = 1
cols = 2

selectors = []
for r in range(rows*cols):
    sel = pn.widgets.Select(name=f'Cell {r+1} model', options=models)
    selectors.append(sel)

plot_panes = [pn.pane.Plotly(height=350, sizing_mode='stretch_width') for _ in range(rows*cols)]


def update_plots(event=None):
    for i, sel in enumerate(selectors):
        model = sel.value
        sub = df[df['model']==model]
        if sub.empty:
            plot_panes[i].object = {}
            continue
        # simple box plot using plotly express
        import plotly.express as px
        fig = px.box(sub, x='k', y='corr', title=f"{model} - corr")
        plot_panes[i].object = fig


for sel in selectors:
    sel.param.watch(update_plots, 'value')

layout = pn.Column(
    pn.Row(*selectors),
    pn.Spacer(height=10),
    pn.Row(*plot_panes)
)

update_plots()

if __name__.startswith('bokeh'):
    pn.serve(layout)

else:
    layout.show()
