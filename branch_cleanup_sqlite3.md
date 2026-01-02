# Branch Cleanup Plan: sqlite3-legacy

This plan outlines the steps to clean the `sqlite3-legacy` branch, removing vector-specific code and files to maintain a pure legacy sqlite3 BLOB-based implementation. The goal is to have a clean repository focused on BLOB storage and O(N²) distance precomputation, without any sql-vector dependencies or code paths.

## Files to Remove
Remove the following files entirely, as they are specific to the sql-vector refactor and not needed in the legacy branch:
- `sqlvector_utils.py` (vector DB utilities)
- `utils_vec.py` (vector-specific helpers)
- `sqlvector_projector.py` (vector projection storage)
- `proj_helpers_vec.py` (vector projection helpers)
- `project_refactor.py` (refactored projection pipeline using vectors)
- `evaluate_clusters_vec.py` (vector-aware evaluation)
- `test_sqlvector_refactor.py` (vector refactor tests)
- `migrate_and_usage.py` (migration scripts for vectors)
- `sql_plotting.py` (vector-specific plotting)

## Files to Keep and Verify
Ensure these legacy files remain and are functional:
- `utils.py` (primary legacy utilities for BLOB embeddings and distances)
- `clustering.py` (needs adaptation: ensure BLOB insertion and O(N²) distances)
- `evaluate_clusters.py` (legacy KNN evaluation, ensure uses load_embeddings/load_distances)
- `evaluate_clusters_vec.py` (remove, as it's vector-specific)
- `cluster_visualization.py` (needs adaptation: ensure uses legacy projection tables)
- `project.py` (legacy projection computation)
- `main.py` (needs adaptation: ensure uses legacy paths, e.g., project.py instead of project_refactor.py)
- `model.py`, `train.py`, `dataset.py` (core model/dataset code, works for both)
- `cfg.py`, `requirements.txt`, `README.md`, `CODEBASE.md` (config and docs, works for both)
- `plotting.py`, `viz_helper.py`, `knn_eval.py`, `knn_panel.py`, `knn_streamlit.py`, `knn_dashboard.py` (plotting and UI, may need minor adaptations but largely compatible)
- `dash_app.py` (dashboard, compatible)
- `test_cocodataset_augmentation_integration.py` (tests, compatible)
- `launch_dino.sh` (script, compatible)

## Code Modifications (Plan Only - Do Not Execute)
While not modifying code in this plan, note that in the legacy branch, future development should:
- Ensure `Clustering_layer.distance_matrix_db` uses BLOB insertion and populates the `distances` table fully.
- Keep `evaluate_clusters.py` loading full matrices for sklearn KNN.
- Use `cluster_visualization.py` with legacy projection tables (`comp_*` columns).
- Remove any imports or conditional logic referencing vector modules.

## Repository Cleanup
- Remove `__pycache__/` directories: `find . -type d -name __pycache__ -exec rm -rf {} +`
- Remove log files: `rm -f job.err job.out wandb/debug*.log`
- Remove wandb artifacts: `rm -rf wandb/`
- Remove temporary or generated files: `rm -f *.pyc` and any debug images like `debug_first_image.png`
- Clean .gitignore: Ensure it includes `*.db`, `wandb/`, `__pycache__/`, etc., but remove vector-specific ignores if any.

## Git Operations
- After cleanup: `git add . && git commit -m "Clean sqlite3-legacy branch: remove vector code"`
- Push: `git push origin sqlite3-legacy`

## Validation
- Run a basic test: `python main.py --cluster --debug` to ensure embedding generation works with BLOB storage.
- Verify KNN evaluation: `python main.py --knn --debug` loads distances correctly.
- Check that no vector-related imports fail (e.g., no `sqlite_vec` errors).

This cleanup results in a streamlined repository for the legacy sqlite3 workflow.