# vllmcluster — Codebase Overview

This document describes the intent and structure of the `vllmcluster`
workspace, focusing on embedding storage, projection pipelines and the
two supported SQLite storage strategies: legacy sqlite3 blobs and the
sql-vector (vec0) virtual-table approach.

Core directories and files
- `vllmcluster/` — primary package for clustering and visualization.
  - `main.py` — CLI entrypoint orchestrating tasks.
  - `clustering.py` — clustering and distance matrix helpers.
  - `model.py`, `train.py`, `dataset.py` — model and dataset utilities.
  - `project_refactor.py` — new pipeline that computes projections using `sqlvector_projector.py`.
  - `sqlvector_projector.py` — implements vec0-backed tables, insertion helpers,
    and kNN helpers (`vec_projections`, `vec_metrics`) using `sqlite_vec` when available.
  - `sqlvector_utils.py` — additional helpers and fallbacks for vec0 usage.
  - `utils.py` — legacy utility helpers for sqlite3-based embeddings (BLOB `embedding` column and per-field tables).
  - `utils_vec.py` — new: sql-vector specific helpers (keeps sql-vector code separate from legacy `utils.py`).
  - `proj_helpers_vec.py` — new: helpers to enumerate runs in proj DBs that use vec_projections/metadata.
  - `cluster_visualization.py` — plotting and wandb logging (legacy code currently writes plain projection tables).
  - `evaluate_clusters.py` — KNN evaluation pipeline; currently reads embeddings and projection tables using legacy assumptions.
  - `knn_streamlit.py` — Streamlit viewer for KNN results (reads `knn_results.db`) — UI unaffected by projection storage choice but depends on evaluation outputs.

Design notes
- Dual-storage compatibility: The repo contains both legacy sqlite3 code paths (BLOBs in `embeddings` tables and plain projection tables with `comp_*` or `x,y` columns) and a modern sql-vector approach (virtual tables `vec_projections` and `vec_metrics`).
- Separation strategy: To make a future branch-based refactor easier we added `utils_vec.py` and `proj_helpers_vec.py` that encapsulate sql-vector-specific behavior. This keeps `utils.py` intact for the sqlite3 legacy branch.

How projections are stored
- Legacy: Per-projection tables such as `umap_<hash>` / `tsne_<hash>` with columns `comp_0, comp_1, ...` or `embeddings` table with `x,y` columns.
- sql-vector: `proj/umap.db` and `proj/tsne.db` contain a `vec_projections` table (virtual or fallback) with an `embedding` column storing a float32 vector per row plus a small `metadata` table mapping `run_id -> params`.

Which files to change when switching branches
- For sqlite3-only branch: keep using `utils.py`, `cluster_visualization.py`, and `evaluate_clusters.py` as-is.
- For sql-vector branch: import and use `utils_vec.py`, `proj_helpers_vec.py`, and `sqlvector_projector.py` APIs instead of the legacy table queries. `project_refactor.py` already uses `sqlvector_projector`.

Next steps / recommendations
- Short-term: Use `utils_vec.py` and `proj_helpers_vec.py` in new sql-vector-aware entrypoints or wrap calls in compatibility shims.
- Medium-term: Replace `cluster_visualization.py` table creation with calls to `sqlvector_projector.compute_and_store` so projection storage is uniform.
- Add `.gitignore` rules for `wandb/`, `job*.err`, `job*.out`, `__pycache__/`, and `*.pyc` to prevent log/noise diffs.

Contact
- If you want, I can apply the compatibility shims to `evaluate_clusters.py` and `cluster_visualization.py` (non-destructive additions) so both storage backends work transparently.  
