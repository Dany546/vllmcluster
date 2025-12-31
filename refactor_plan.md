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


