# Branch Cleanup Plan: sql-vector-refactor

This plan outlines the steps to clean the `sql-vector-refactor` branch, removing legacy sqlite3 code and files to maintain a pure vector-enabled SQLite implementation. The goal is to have a clean repository focused on vector storage, indexing, and on-demand kNN, without legacy BLOB dependencies.

## Files to Remove
Remove the following files entirely, as they are specific to the legacy sqlite3 implementation and not needed in the vector branch:
- `utils.py` (legacy BLOB utilities; replaced by `utils_vec.py`)
- `project.py` (legacy projection computation; replaced by `project_refactor.py`)
- `migrate_and_usage.py` (legacy migration scripts)
- `sql_plotting.py` (legacy-specific plotting)
- Temporary or debug files: `debug_first_image.png`, etc.

## Files to Keep and Verify
Ensure these vector-specific files remain and are functional:
- `sqlvector_utils.py` (core vector DB utilities)
- `utils_vec.py` (vector-specific helpers)
- `sqlvector_projector.py` (vector projection storage and kNN)
- `proj_helpers_vec.py` (vector projection enumeration)
- `project_refactor.py` (refactored projection pipeline)
- `clustering.py` (needs adaptation: ensure vector insertion and indexing, remove O(N²) distances)
- `evaluate_clusters_vec.py` (vector-aware evaluation, already adapted)
- `evaluate_clusters.py` (remove or keep as backup, but use _vec version)
- `cluster_visualization.py` (needs adaptation: ensure loads from `vec_projections`)
- `main.py` (needs adaptation: ensure uses vector paths, e.g., project_refactor.py)
- `model.py`, `train.py`, `dataset.py` (core model/dataset code, works for both)
- `cfg.py`, `requirements.txt`, `README.md`, `CODEBASE.md` (config and docs, works for both)
- `plotting.py`, `viz_helper.py`, `knn_eval.py`, `knn_panel.py`, `knn_streamlit.py`, `knn_dashboard.py` (plotting and UI, may need minor adaptations but largely compatible)
- `dash_app.py` (dashboard, compatible)
- `test_cocodataset_augmentation_integration.py` (tests, compatible)
- `launch_dino.sh` (script, compatible)
- `evaluate_clusters_vec.py` (vector-aware evaluation, keep as alternative or merge)
- `refactor_plan.md` (updated plan)

## Code Modifications (Plan Only - Do Not Execute)
While not modifying code in this plan, note that in the vector branch, future development should:
- Ensure `Clustering_layer.distance_matrix_db` inserts into vector tables and builds indexes, removing O(N²) distances.
- Update `evaluate_clusters.py` to query `query_knn` directly from DB for KNN, avoiding memory-intensive matrix loads.
- Use `project_refactor.py` and `sqlvector_projector.py` for projections.
- Add fallbacks for when `sqlite_vec` is unavailable.
- Remove legacy imports and conditional logic for BLOB paths.

## Repository Cleanup
- Remove `__pycache__/` directories: `find . -type d -name __pycache__ -exec rm -rf {} +`
- Remove log files: `rm -f job.err job.out wandb/debug*.log`
- Remove wandb artifacts: `rm -rf wandb/`
- Remove temporary or generated files: `rm -f *.pyc` and any legacy debug files
- Clean .gitignore: Ensure it includes `*.db`, `wandb/`, `__pycache__/`, and add vector-specific ignores if needed (e.g., extension binaries).

## Git Operations
- After cleanup: `git add . && git commit -m "Clean sql-vector-refactor branch: remove legacy code"`
- Push: `git push origin sql-vector-refactor`

## Validation
- Run a basic test: `python main.py --cluster --debug` to ensure embedding generation works with vector storage and indexing.
- Verify KNN evaluation: `python main.py --knn --debug` uses DB queries for kNN.
- Check that vector extension loads correctly (or falls back gracefully).
- Ensure no legacy BLOB-related errors (e.g., no calls to `load_distances` for full matrices).

This cleanup results in a streamlined repository for the sql-vector workflow.