# Refactor Plan — Vectorized SQLite Integration

## Goal
- Replace raw BLOB embedding storage and O(N²) precomputed distances with a local **vector‑enabled SQLite** workflow.
- Regenerate embeddings into a typed `VECTOR(dim)` column, **build a persistent index once**, and use **on‑demand kNN** queries for training/evaluation.
- Preserve metadata and prediction logging; enable fast, repeatable kNN experiments (vary `k` and weighting) and reproducible CV logging.

---

## Current state
- **Embeddings**: generated and stored as raw BLOBs (`vec.numpy().tobytes()`) in a SQLite `embeddings` table.
- **Predictions & distances**: `predictions` table exists; `distances` table is populated by an O(N²) `torch.cdist` loop in `Clustering_layer.distance_matrix_db`.
- **Model pipeline**: `yolowrapper` produces embeddings and per‑image outputs; `distance_matrix_db` handles insertion and full pairwise distance computation.
- **Refactor prototype**: helper design (`sqlvector_utils.py`) and kNN evaluation helpers sketched; insertion logic updated to target a `VECTOR(dim)` column and to call an index builder, but **extension API, serialization format, and exact SQL** are not finalized.
- **Migration**: not required — you will regenerate embeddings from scratch.

---

## Left to do
1. **Install & configure vector extension**
   - Obtain the local SQLite vector extension binary for your platform and set `VECTOR_EXT_PATH` (or ensure `conn.load_extension` can find it).

2. **Finalize `sqlvector_utils`**
   - Adapt `build_vector_index` and `query_knn` SQL to the extension’s API (operator/function names like `<->`, `vector_distance`, or `CREATE INDEX ... USING hnsw(...)`).
   - Confirm whether the extension expects raw `float32` bytes or Python lists and update `serialize_float32_array` accordingly.

3. **Align insertion with model outputs**
   - Verify `match_targets_to_preds` / `compute_losses` output shapes and ensure the insertion code extracts losses/metrics consistently.
   - Convert tensors to NumPy (`.cpu().numpy()`) before serializing/storing.

4. **Replace distance usage**
   - Remove the O(N²) `torch.cdist` block (or keep `distances` as an empty compatibility stub).
   - Update downstream code to use `query_knn` for on‑demand neighbor retrieval.

5. **Reproducibility & metadata**
   - Record index build parameters (seed, `efConstruction`, `M`, etc.) and `total_count` (as integer) in `metadata`.

6. **DB usage & performance**
   - Reuse connections, batch inserts, and commit in larger batches for throughput.
   - Build the index once after all inserts; tune index params for recall/performance.

7. **Validation & tests**
   - **Extension smoke test**: insert a few synthetic vectors, build index, run `query_knn`.
   - **Round‑trip test**: compare index neighbors vs exact `torch.cdist` on a held subset.
   - **Loss integrity test**: insert a YOLO batch and verify stored loss values match `compute_losses`.
   - **End‑to‑end CV test**: run one fold, compare predictions and metrics to in‑memory baseline.
   - **Reproducibility test**: rebuild index with same params and verify stability (or record nondeterminism).

---

## Minimal immediate checklist
- [ ] Place vector extension binary and set `VECTOR_EXT_PATH`.
- [ ] Add `sqlvector_utils.py` and adapt SQL to your extension.
- [ ] Update `Clustering_layer.distance_matrix_db` to insert vector rows and call `build_vector_index`.
- [ ] Remove full pairwise `torch.cdist` block and switch evaluation to `query_knn`.
- [ ] Run smoke, round‑trip, CV, and reproducibility tests.

---

## Notes
- For **~5k** embeddings the index build is fast; the main benefit is amortized speed when running many folds or hyperparameter sweeps.
- Validate approximate index recall vs exact neighbors before relying on weight‑sensitive kNN variants.
- Keep `distances` table only if a downstream algorithm explicitly requires persisted all‑pairs distances.

---

## Branching Strategy for Refactor
To cleanly separate the legacy sqlite3 functionality from the new sql-vector approach, we will create two git branches from the current state. This allows parallel development and testing without conflicts.

### Branch 1: `sqlite3-legacy`
- **Purpose**: Maintain and stabilize the current BLOB-based embedding storage and O(N²) distance precomputation.
- **Key Files to Keep/Modify**:
  - Retain `utils.py` as the primary utility module.
  - Keep `Clustering_layer.distance_matrix_db` with the O(N²) `torch.cdist` loop and BLOB insertion.
  - Ensure `evaluate_clusters.py` uses `load_embeddings` and `load_distances` from `utils.py` for full matrix KNN.
  - Use `cluster_visualization.py` and `project.py` (legacy projection tables with `comp_*` or `x,y` columns).
  - Remove or disable vector-specific imports (e.g., `sqlvector_utils.py`, `utils_vec.py`).
- **Changes**:
  - Comment out or remove calls to `build_vector_index` and vector table creation.
  - Ensure `distances` table is fully populated.
  - Update any conditional logic to default to legacy paths.
- **Testing**: Verify KNN evaluation works with precomputed distances; ensure projections load correctly from legacy tables.

### Branch 2: `sql-vector-refactor`
- **Purpose**: Implement and optimize the vector-enabled SQLite workflow with on-demand kNN.
- **Key Files to Modify/Integrate**:
  - Integrate `sqlvector_utils.py`, `utils_vec.py`, `sqlvector_projector.py`, and `proj_helpers_vec.py`.
  - Update `Clustering_layer.distance_matrix_db` to use vector insertion and index building; remove O(N²) distance computation.
  - Refactor `evaluate_clusters.py` to use `query_knn` from `sqlvector_utils.py` instead of loading full matrices (implement DB-based KNN to avoid memory issues).
  - Update `cluster_visualization.py` to load from `vec_projections` tables.
  - Use `project_refactor.py` for projections.
- **Changes**:
  - Implement the "Left to do" checklist above.
  - Add compatibility shims in `evaluate_clusters.py` to query vector DBs directly for KNN (e.g., replace sklearn KNN with DB queries).
  - Ensure fallbacks work when extension is unavailable.
  - Update `main.py` to conditionally use vector paths if extension is detected.
- **Testing**: Run the validation tests (smoke, round-trip, CV, reproducibility); compare performance against legacy branch.

### Implementation Steps for Branching
1. **Create Branches**:
   - From current `main` branch: `git checkout -b sqlite3-legacy`
   - Then: `git checkout main; git checkout -b sql-vector-refactor`

2. **Isolate Changes**:
   - In `sqlite3-legacy`: Remove vector-related files or make them no-ops; ensure legacy code paths are active.
   - In `sql-vector-refactor`: Implement the refactor plan; add new files and update existing ones to use vector APIs.

3. **Merge Strategy**:
   - Develop independently; periodically merge bug fixes from `sqlite3-legacy` to `sql-vector-refactor`.
   - Once `sql-vector-refactor` is stable, consider merging back or keeping as feature branch.

4. **Risks & Rollback**:
   - If vector extension issues arise, rollback to `sqlite3-legacy` for production.
   - Ensure DB schemas are versioned to avoid corruption.

This plan ensures the refactor is incremental, testable, and reversible.
