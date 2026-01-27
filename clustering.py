import logging
import sqlite3
from pydoc import classify_class_attrs
import io
import json
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from cfg import cat_to_super, device
from model import CLIP, DINO
from typing import List
from typing_extensions import Optional, Tuple
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from types import SimpleNamespace
from yolo_extract import YOLOExtractor


def box_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (torch.Tensor): First bounding box, shape [4] (x1,y1,x2,y2)
        box2 (torch.Tensor): Second bounding box, shape [4] (x1,y1,x2,y2)

    Returns:
        torch.Tensor: IoU value, scalar
    """
    box1 = box1.reshape(-1, 4)
    box2 = box2.reshape(-1, 4)
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou


def compute_dice_score(pred_masks, gt_masks):
    """
    Compute Dice score between predicted and ground truth masks.
    
    Args:
        pred_masks: Predicted masks [N, H, W]
        gt_masks: Ground truth masks [N, H, W]
    
    Returns:
        Dice score (scalar)
    """
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return torch.tensor(0.0)
    
    pred_masks = (pred_masks > 0.5).float()
    gt_masks = (gt_masks > 0.5).float()
    
    intersection = (pred_masks * gt_masks).sum(dim=(1, 2))
    union = pred_masks.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2))
    
    dice = (2.0 * intersection) / (union + 1e-6)
    return dice.mean()


# --- helper utilities for eval fallbacks ---

def _class_jaccard(classes_a, classes_b):
    """Return Jaccard (intersection/union) between two class lists/tensors.

    Both inputs may be torch tensors or Python iterables. Empty==empty -> 1.0.
    """
    if isinstance(classes_a, torch.Tensor):
        a = set([int(x) for x in classes_a.tolist()])
    else:
        a = set([int(x) for x in classes_a]) if classes_a is not None else set()
    if isinstance(classes_b, torch.Tensor):
        b = set([int(x) for x in classes_b.tolist()])
    else:
        b = set([int(x) for x in classes_b]) if classes_b is not None else set()
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _agg_class_masks(pred, device=torch.device('cpu')):
    """Aggregate per-instance masks into per-class masks (dict: class_id -> mask tensor).

    pred: dict with keys 'masks' and 'classes'. 'masks' may be a tensor of shape [N, H, W]
    or a list/iterable of masks. Returns an empty dict when no masks present.
    """
    m = pred.get('masks', None)
    if m is None or getattr(m, 'numel', lambda:0)() == 0:
        return {}
    masks = m
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks, dtype=torch.float32, device=device)
    else:
        masks = masks.to(device)
    classes = pred.get('classes', torch.tensor([], dtype=torch.long))
    cls_list = [int(x) for x in classes.tolist()] if torch.is_tensor(classes) else list(classes)
    agg = {}
    for idx, cid in enumerate(cls_list):
        cid = int(cid)
        mask = masks[idx]
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if cid in agg:
            agg[cid] = agg[cid] | mask
        else:
            agg[cid] = mask.clone()
    return agg


def _dice_mask(a, b):
    """Dice coefficient for two boolean masks (same shape). Returns float in [0,1]."""
    if a is None or b is None:
        return 0.0
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a_f = a.float() if a.dtype != torch.float32 else a
    b_f = b.float() if b.dtype != torch.float32 else b
    inter = (a_f * b_f).sum()
    sum_a = a_f.sum()
    sum_b = b_f.sum()
    if float(sum_a) == 0.0 and float(sum_b) == 0.0:
        return 1.0
    return float((2.0 * inter) / (sum_a + sum_b + 1e-6))


def _box_similarity(boxes_i, boxes_j, device=torch.device('cpu')):
    """Set-to-set box similarity between two images' boxes. Returns scalar in [0,1].

    Approach: compute pairwise IoU, then take mean of per-box best matches (both directions) and average.
    """
    if boxes_i is None or boxes_j is None:
        return 0.0
    if (not torch.is_tensor(boxes_i) or boxes_i.numel() == 0) and (not torch.is_tensor(boxes_j) or boxes_j.numel() == 0):
        return 1.0
    if not torch.is_tensor(boxes_i):
        boxes_i = torch.tensor(boxes_i, device=device, dtype=torch.float32)
    else:
        boxes_i = boxes_i.to(device).float()
    if not torch.is_tensor(boxes_j):
        boxes_j = torch.tensor(boxes_j, device=device, dtype=torch.float32)
    else:
        boxes_j = boxes_j.to(device).float()
    if boxes_i.numel() == 0 or boxes_j.numel() == 0:
        return 0.0
    # pairwise IoU
    a_x1 = boxes_i[:, 0].unsqueeze(1)
    a_y1 = boxes_i[:, 1].unsqueeze(1)
    a_x2 = boxes_i[:, 2].unsqueeze(1)
    a_y2 = boxes_i[:, 3].unsqueeze(1)
    b_x1 = boxes_j[:, 0].unsqueeze(0)
    b_y1 = boxes_j[:, 1].unsqueeze(0)
    b_x2 = boxes_j[:, 2].unsqueeze(0)
    b_y2 = boxes_j[:, 3].unsqueeze(0)
    ix1 = torch.max(a_x1, b_x1)
    iy1 = torch.max(a_y1, b_y1)
    ix2 = torch.min(a_x2, b_x2)
    iy2 = torch.min(a_y2, b_y2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_a = ((boxes_i[:, 2] - boxes_i[:, 0]) * (boxes_i[:, 3] - boxes_i[:, 1])).unsqueeze(1)
    area_b = ((boxes_j[:, 2] - boxes_j[:, 0]) * (boxes_j[:, 3] - boxes_j[:, 1])).unsqueeze(0)
    union = area_a + area_b - inter + 1e-6
    iou = inter / union
    a_to_b = iou.max(dim=1).values.mean()
    b_to_a = iou.max(dim=0).values.mean()
    return float((a_to_b + b_to_a) / 2.0)

def array_to_blob(array: np.ndarray) -> bytes:
    """Serialize a numpy array to raw bytes (float32).

    This is suitable for storing fixed-shape float arrays in a BLOB column.
    """
    return array.astype(np.float32).tobytes()


def blob_to_array(blob: bytes, dtype=np.float32) -> np.ndarray:
    """Deserialize raw bytes back to a numpy array with given dtype."""
    return np.frombuffer(blob, dtype=dtype)


def tensors_to_heads_blob(tensors: List[torch.Tensor], compress: bool = True):
    """Serialize a list of tensors (heads) into a compressed binary blob
    and return metadata (shapes, dtypes).

    Returns
    -------
    blob: bytes
        Compressed binary blob containing all heads (npz)
    shapes_json: str
        JSON list of shapes for each head
    dtypes_json: str
        JSON list of dtypes for each head
    """
    np_arrays = []
    shapes = []
    dtypes = []
    for t in tensors:
        a = t.detach().cpu().numpy()
        np_arrays.append(a.astype(np.float32))
        shapes.append(list(a.shape))
        dtypes.append(str(a.dtype))
    buf = io.BytesIO()
    savez_kwargs = {f"h{i}": arr for i, arr in enumerate(np_arrays)}
    if compress:
        np.savez_compressed(buf, **savez_kwargs)
    else:
        np.savez(buf, **savez_kwargs)
    blob = buf.getvalue()
    buf.close()
    return blob, json.dumps(shapes), json.dumps(dtypes)


def heads_blob_to_tensors(blob: bytes) -> List[torch.Tensor]:
    """Deserialize a compressed heads blob back to a list of tensors."""
    buf = io.BytesIO(blob)
    with np.load(buf, allow_pickle=False) as npz:
        arrays = [npz[f] for f in npz.files]
    return [torch.from_numpy(a) for a in arrays]


class Clustering_layer:
    def __init__(self, model_str, model_name, debug=False, store_individual_predictions: bool = False):
        self.model_str = model_str
        self.model_name = model_name
        # Control whether per-detection predictions (including masks) are stored in the DB
        self.store_individual_predictions = store_individual_predictions
        
        attention_pooling = "attention" in model_str
        v3 = "v3" in model_str
        model_str = model_str.replace("_attention", "").replace("v3", "")
        
        # Check if segmentation model
        self.is_seg = 'seg' in model_str.lower()
        
        if model_str == "dino":
            self.model = DINO(attention_pooling=attention_pooling).to(device)
        elif model_str == "clip":
            self.model = CLIP().to(device)
        elif "yolo" in model_str:
            self.model = YOLOExtractor(model_str, device=device)
        else:
            raise ValueError(f"Unknown model string: {model_str}")

        self.debug = debug
        logging.basicConfig(
            format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.embeddings_db = (
            f"/CECI/home/ucl/irec/darimez/embeddings/{model_name}.db"
        )
        self.distances_db = (
            f"/CECI/home/ucl/irec/darimez/distances/{model_name}.db"
        )

    def _infer_resume_blocks(self, dist_cursor, emb_cursor, batch_size, n_blocks, n_samples):
        """Return (start_i_block, start_j_block) to resume distance computation.

        Logic:
        - Prefer progress table if present (resumes after last_j).
        - Otherwise scan i_blocks in increasing order and find the first i_block
          that is missing at least one j_block; resume at that j_block.
        - If no incomplete block is found, fall back to last block (start_j=0).
        """
        # Progress table takes precedence
        dist_cursor.execute("SELECT last_i, last_j FROM progress ORDER BY last_i DESC LIMIT 1")
        progress_row = dist_cursor.fetchone()
        if progress_row:
            resume_i, resume_j = progress_row
            start_i_block = int(resume_i)
            start_j_block = int(resume_j) + 1
            if start_j_block >= n_blocks:
                start_i_block += 1
                start_j_block = 0
            if start_i_block >= n_blocks:
                start_i_block = max(0, n_blocks - 1)
                start_j_block = 0
            self.logger.info(
                f"Resuming distance computation from progress table: i_block={start_i_block}, j_block={start_j_block}"
            )
            return start_i_block, start_j_block

        # No explicit progress checkpoint. Inspect distances table for incomplete blocks
        dist_cursor.execute("SELECT COUNT(*) FROM distances")
        (dist_count_raw,) = dist_cursor.fetchone()
        dist_count = int(dist_count_raw) if dist_count_raw is not None else 0
        if dist_count == 0:
            return 0, 0

        for ib in range(0, n_blocks):
            i_start = int(ib * batch_size)
            i_end = int(min((ib + 1) * batch_size, n_samples))
            dist_cursor.execute("SELECT DISTINCT j FROM distances WHERE i >= ? AND i < ?", (i_start, i_end))
            present_j_ids = [r[0] for r in dist_cursor.fetchall()]
            present_j_blocks = {int(jid) // batch_size for jid in present_j_ids}

            missing_j_block = None
            for b in range(0, n_blocks):
                if b not in present_j_blocks:
                    missing_j_block = b
                    break

            if missing_j_block is not None:
                self.logger.info(
                    f"Distances DB already contains {dist_count} rows; resuming from incomplete i_block={ib}, j_block={missing_j_block}"
                )
                return ib, missing_j_block

        # All blocks appear complete — fall back to last block
        start_i_block = max(0, n_blocks - 1)
        start_j_block = 0
        self.logger.info(f"Distances DB contains {dist_count} rows but no incomplete blocks found; resuming at block i_block={start_i_block}, j_block={start_j_block}")
        return start_i_block, start_j_block

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def distance_matrix_db(
        self,
        data_loader,
        pragma_speed: bool = True,
        eval_mode: bool = False,
        eval_on_gpu: bool = False,
    ) -> Tuple[str, str]:
        """Compute pairwise distances and store embeddings in a vector-enabled SQLite DB."""
        # try to read declared batch size from the dataloader, fallback to inspecting first batch
        batch_size = getattr(data_loader, "batch_size", None)
        if batch_size is None:
            batch_size = len(next(iter(data_loader))[0])
        embeddings_db = self.embeddings_db
        distances_db = self.distances_db  # you can keep this path if you want to store other info
        # If user requested evaluation-mode distances, write to a separate DB (append '-eval' to filename)
        if eval_mode:
            base, ext = os.path.splitext(distances_db)
            if not base.endswith("-eval"):
                distances_db = f"{base}-eval{ext}"
                # Persist the chosen eval distances DB path on the object so tests/clients can find it
                self.distances_db = distances_db
        # --- Open vector-enabled embeddings DB ---
        from sqlvector_utils import create_embeddings_table, serialize_float32_array, insert_embeddings_batch, build_vector_index

        # Use sqlite3 directly (no APSW / sqlvector refactor) to ensure compatibility
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        emb_cursor = emb_conn.cursor()
        # dimension: infer from model output by running one batch or set explicitly
        # Here we infer from a dummy forward pass if possible
        # We'll assume model returns embeddings of shape [B, D] when called with images
        # Dimension will be inferred later only if needed. Avoid running model predictor here to prevent
        # repeating expensive inference when embeddings are already present from a previous run.
        dim = None
        # For non-YOLO models, ensure the embeddings table schema exists (dim is only needed when creating it)
        emb_cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                img_id INTEGER,
                embedding BLOB,
                hit_freq REAL,
                mean_iou REAL,
                mean_conf REAL,
                flag_cat INTEGER,
                flag_supercat INTEGER
            )
        """)
        
        emb_cursor.execute(
            """CREATE TABLE IF NOT EXISTS predictions_raw_heads (
                id INTEGER PRIMARY KEY,
                img_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                head_blob BLOB NOT NULL,
                head_shapes TEXT NOT NULL,
                head_dtypes TEXT NOT NULL,
                anchors_info TEXT,
                features BLOB,
                created_ts INTEGER,
                UNIQUE(img_id, model_name)
            )"""
        )
        
        # Features table for storing per-image GAP and bottleneck features
        emb_cursor.execute(
            """CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY,
                img_id INTEGER NOT NULL,
                gap_features BLOB,
                bottleneck_features BLOB,
                created_ts INTEGER,
                UNIQUE(img_id)
            )"""
        )
        
        # Detailed predictions table for storing individual detections (optional)
        if "yolo" in self.model_name and self.store_individual_predictions:
            emb_cursor.execute(
                """CREATE TABLE IF NOT EXISTS predictions_individual (
                    id INTEGER PRIMARY KEY,
                    img_id INTEGER NOT NULL,
                    box_x1 REAL,
                    box_y1 REAL,
                    box_x2 REAL,
                    box_y2 REAL,
                    confidence REAL,
                    class_id INTEGER,
                    class_name TEXT,
                    mask BLOB,
                    created_ts INTEGER
                )"""
            )
        
        emb_cursor.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value INTEGER)"
        )
        emb_conn.commit()
        # Embedding dimension `dim` will be determined after checking for existing embeddings (to avoid unnecessary inference).
        # Ensure 'predictions' table exists with expected schema (create if missing)
        try:
            emb_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
            if not emb_cursor.fetchone():
                if self.is_seg:
                    emb_cursor.execute(
                        """CREATE TABLE IF NOT EXISTS predictions (
                            id INTEGER PRIMARY KEY,
                            img_id INTEGER,
                            iou BLOB,
                            conf BLOB,
                            dice BLOB,
                            cat BLOB,
                            supercat BLOB
                        )"""
                    )
                else:
                    emb_cursor.execute(
                        """CREATE TABLE IF NOT EXISTS predictions (
                            id INTEGER PRIMARY KEY,
                            img_id INTEGER,
                            iou BLOB,
                            conf BLOB,
                            cat BLOB,
                            supercat BLOB
                        )"""
                    )
                emb_conn.commit()

            # Now validate and add any missing columns (non-destructive)
            emb_cursor.execute("PRAGMA table_info(predictions)")
            existing_cols = [r[1] for r in emb_cursor.fetchall()]
            required = ['id', 'img_id', 'iou', 'conf', 'cat', 'supercat']
            if self.is_seg and 'dice' not in existing_cols:
                required.insert(4, 'dice')
            for c in required:
                if c not in existing_cols:
                    if c == 'id':
                        self.logger.warning("Existing 'predictions' table missing 'id' column; cannot auto-add primary key")
                    else:
                        emb_cursor.execute(f"ALTER TABLE predictions ADD COLUMN {c} BLOB")
            emb_conn.commit()
        except Exception as e:
            self.logger.debug(f"Could not validate or migrate 'predictions' table: {e}")

        # Removed redundant early embedding pass. Embedding extraction and feature/prediction saving
        # are performed in the canonical 'Step 1: Check and compute embeddings' pass below.

        # Optionally keep distances DB for other uses, but we no longer precompute all pairwise distances here.
        # If you still need a distances DB for compatibility, you can keep creating it but leave it empty or compute on demand.
        # Open distances DB with a timeout so write attempts wait instead of failing immediately
        dist_conn = sqlite3.connect(distances_db, timeout=30)
        # Make SQLite wait up to 30s when the DB is locked (helps transient writer contention)
        dist_conn.execute("PRAGMA busy_timeout = 30000")  # milliseconds

        # We avoid forcing journal mode changes here (other processes may hold locks).
        # Set a few non-blocking performance PRAGMAs where applicable (may be no-ops depending on mode)
        dist_conn.execute("PRAGMA synchronous = OFF")
        dist_conn.execute("PRAGMA temp_store = MEMORY")
        dist_conn.execute("PRAGMA cache_size = -1024000")  # 1GB cache hint
        dist_conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O
        dist_conn.execute("PRAGMA automatic_index = OFF")  # Prevent auto-index during inserts

        dist_cursor = dist_conn.cursor()

        # Helper utilities to retry writes/commits transiently when DB is locked
        def _exec_with_retry(cursor, sql, params=None, retries=5, backoff=0.5):
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    if params is None:
                        return cursor.execute(sql)
                    else:
                        return cursor.executemany(sql, params) if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], (list, tuple)) else cursor.execute(sql, params)
                except sqlite3.OperationalError as e:
                    last_err = e
                    self.logger.warning(f"DB write attempt {attempt}/{retries} failed (OperationalError: {e}); retrying after {backoff*attempt:.1f}s")
                    time.sleep(backoff * attempt)
            # Exhausted retries — re-raise the last error to show full traceback
            raise last_err

        def _commit_with_retry(conn, retries=5, backoff=0.5):
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    conn.commit()
                    return
                except sqlite3.OperationalError as e:
                    last_err = e
                    self.logger.warning(f"DB commit attempt {attempt}/{retries} failed (OperationalError: {e}); retrying after {backoff*attempt:.1f}s")
                    time.sleep(backoff * attempt)
            raise last_err
        # Ensure distances and progress tables exist; use retry helpers to avoid transient lock failures
        try:
            _exec_with_retry(dist_cursor, """
                CREATE TABLE IF NOT EXISTS distances (
                    i INTEGER,
                    j INTEGER,
                    component TEXT,
                    distance REAL,
                    PRIMARY KEY (i, j, component)
                )
            """)
            _exec_with_retry(dist_cursor, "CREATE TABLE IF NOT EXISTS progress (last_i INTEGER PRIMARY KEY, last_j INTEGER)")

            # Create indices for the component-aware distances table (idempotent)
            _exec_with_retry(dist_cursor, "CREATE INDEX IF NOT EXISTS idx_i ON distances(i)")
            _exec_with_retry(dist_cursor, "CREATE INDEX IF NOT EXISTS idx_j ON distances(j)")
            _exec_with_retry(dist_cursor, "CREATE INDEX IF NOT EXISTS idx_component ON distances(component)")
            _commit_with_retry(dist_conn)
        except Exception as e:
            # Log and proceed; the tables may be managed externally or creation may fail due to locks
            self.logger.warning(f"Could not ensure distances/progress tables exist: {e}")
            # Continue — _infer_resume_blocks will handle missing tables gracefully (fallback below)

        # Step 1: Check and compute embeddings
        # Ensure embeddings DB is open (we closed it earlier after building index)
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        emb_cursor = emb_conn.cursor()
        emb_cursor.execute('SELECT value FROM metadata WHERE key = "total_count"')
        result = emb_cursor.fetchone()
        metadata_count = int(result[0]) if result else 0

        # Verify metadata against actual embeddings table to avoid resuming from stale metadata
        emb_cursor.execute('SELECT COUNT(*) FROM embeddings')
        actual_count = int(emb_cursor.fetchone()[0])

        if metadata_count != actual_count:
            # Prefer the actual table count (more reliable) and fix metadata
            self.logger.warning(f"Metadata total_count={metadata_count} inconsistent with actual embeddings count={actual_count}. Using actual count and repairing metadata.")
            metadata_count = actual_count
            emb_cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', ("total_count", str(metadata_count)))
            emb_conn.commit()

        if metadata_count == 0:
            self.logger.info("Computing embeddings from scratch...")
        else:
            self.logger.info(f"Found {metadata_count} existing embeddings ...")

        idx = int(metadata_count)
        new_embeddings = 0

        # Determine embedding dimension `dim` now that we can inspect the embeddings DB
        if idx > 0:
            emb_cursor.execute("SELECT embedding FROM embeddings LIMIT 1")
            row = emb_cursor.fetchone()
            if row and row[0]:
                arr = np.frombuffer(row[0], dtype=np.float32)
                dim = int(arr.shape[0])
                self.logger.info(f"Inferred embedding dim={dim} from existing embeddings")
            else:
                dim = 512
                self.logger.warning("Could not infer dim from DB, defaulting to 512")
        else:
            # Need to infer dim by running a single sample through the model (only once)
            self.logger.info("Inferring embedding dimension from a sample batch (single run)")
            img_ids, sample_images, labels = next(iter(data_loader))
            sample_images = sample_images.to(device)
            if "yolo" in self.model_name:
                targets = self.model.labels_to_targets(labels, sample_images.shape[0]) if labels else None
                result = self.model.run_with_predictor(
                    sample_images,
                    targets=targets,
                    conf=0.25,
                    iou=0.45,
                    embed_layers=[-2],
                )
                gap_features = result.get('gap_features', {})
                gap_list = [gap_features[k] for k in sorted(gap_features.keys())]
                sample_emb = torch.cat(gap_list, dim=1) if gap_list else torch.zeros(sample_images.shape[0], 512, device=device)
                dim = sample_emb.shape[1]
            else:
                emb_sample = self.model(sample_images)
                dim = emb_sample.shape[1]

        # Now ensure embeddings table exists with proper dim
        create_embeddings_table(emb_conn, dim, is_seg=self.is_seg)

        # Ensure embeddings table contains the expected columns (add if missing). This handles older DBs
        emb_cursor.execute("PRAGMA table_info(embeddings)")
        emb_cols = {r[1] for r in emb_cursor.fetchall()}
        # Columns we may later insert into
        expected_cols = {
            'mean_dice', 'box_loss', 'cls_loss', 'dfl_loss', 'seg_loss'
        }
        for col in expected_cols:
            if col not in emb_cols:
                emb_cursor.execute(f"ALTER TABLE embeddings ADD COLUMN {col} REAL DEFAULT 0.0")
                self.logger.info(f"Added missing column '{col}' to embeddings table")
        emb_conn.commit()

        # Only compute embeddings if not all are done
        if idx < len(data_loader.dataset):
            self.logger.info(f"Computing remaining {len(data_loader.dataset) - idx} embeddings...")
            self.logger.info(f"Using {self.model_name} model for embedding extraction")
            # If the dataloader supports a start index, set it to the number of already-present embeddings
            # (some dataloader implementations respect `start_idx` to resume iteration cheaply)
            if hasattr(data_loader, 'start_idx'):
                data_loader.start_idx = idx
            for batch_ids, (img_ids, images, labels) in enumerate(data_loader):
                images = images.to(device)
                if "yolo" in self.model_name:
                    # Use new predictor-based approach for YOLO models
                    targets = self.model.labels_to_targets(labels, images.shape[0]) if labels else None
                    result = self.model.run_with_predictor(
                        images,
                        targets=targets,
                        conf=0.25,
                        iou=0.45,
                        embed_layers=[-2, -1] if self.debug else [-2],
                    )
                    
                    # Extract structured predictions, losses, and features
                    structured_preds = result['predictions']
                    losses = result['losses']
                    gap_features = result['gap_features']
                    
                    # Compute synthetic embeddings from GAP features
                    gap_list = [gap_features[k] for k in sorted(gap_features.keys())]
                    if gap_list:
                        emb = torch.cat(gap_list, dim=1)
                    else:
                        emb = torch.zeros(len(structured_preds), 512, device=device)
                    
                    # Process predictions and losses
                    self.model.debug = False # self.debug
                    outputs, all_output = self.model.process_yolo_batch(
                        structured_preds, targets, labels, losses, emb, cat_to_super=cat_to_super
                    )
                else:
                    emb = self.model(images)
                    outputs = None
                    all_output = None

                if "yolo" in self.model_name and outputs is not None:
                    row = []
                    all_row = []
                    
                    if self.is_seg:
                        # Segmentation model: include Dice and seg_loss
                        for id, (img_id, vec, output) in enumerate(
                            zip(img_ids, emb.cpu(), outputs)
                        ):
                            miou, mconf, cat, supercat, hit_freq, dice, box_loss, cls_loss, dfl_loss, seg_loss = output
                            row.append([
                                int((batch_ids * batch_size) + id),
                                int(img_id),
                                vec.numpy().tobytes(),
                                hit_freq,
                                miou,
                                mconf,
                                dice,
                                cat,
                                supercat,
                                box_loss,
                                cls_loss,
                                dfl_loss,
                                seg_loss,
                            ])
                        emb_cursor.executemany(
                            "INSERT OR IGNORE INTO embeddings (id, img_id, embedding, hit_freq, mean_iou, mean_conf, mean_dice, flag_cat, flag_supercat, box_loss, cls_loss, dfl_loss, seg_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            row,
                        )
                        for id, (img_id, miou, mconf, cat, supercat, dice) in enumerate(
                            all_output
                        ):
                            all_row.append([
                                int(batch_ids * batch_size + id),
                                int(img_id),
                                array_to_blob(np.array(miou, dtype=np.float32)),
                                array_to_blob(np.array(mconf, dtype=np.float32)),
                                array_to_blob(np.array(dice, dtype=np.float32)),
                                array_to_blob(np.array(cat, dtype=np.int32)),
                                array_to_blob(np.array(supercat, dtype=np.int32)),
                            ])
                        emb_cursor.executemany(
                            "INSERT OR IGNORE INTO predictions (id, img_id, iou, conf, dice, cat, supercat) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            all_row,
                        )
                    else:
                        # Detection model: no Dice or seg_loss
                        for id, (img_id, vec, output) in enumerate(
                            zip(img_ids, emb.cpu(), outputs)
                        ):
                            miou, mconf, cat, supercat, hit_freq, dice, box_loss, cls_loss, dfl_loss = output
                            row.append([
                                int((batch_ids * batch_size) + id),
                                int(img_id),
                                vec.numpy().tobytes(),
                                hit_freq,
                                miou,
                                mconf,
                                cat,
                                supercat,
                                box_loss,
                                cls_loss,
                                dfl_loss,
                            ])
                        emb_cursor.executemany(
                            "INSERT OR IGNORE INTO embeddings (id, img_id, embedding, hit_freq, mean_iou, mean_conf, flag_cat, flag_supercat, box_loss, cls_loss, dfl_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            row,
                        )
                        for id, (img_id, miou, mconf, cat, supercat, dice) in enumerate(
                            all_output
                        ):
                            all_row.append([
                                int(batch_ids * batch_size + id),
                                int(img_id),
                                array_to_blob(np.array(miou, dtype=np.float32)),
                                array_to_blob(np.array(mconf, dtype=np.float32)),
                                array_to_blob(np.array(cat, dtype=np.int32)),
                                array_to_blob(np.array(supercat, dtype=np.int32)),
                            ])
                        emb_cursor.executemany(
                            "INSERT OR IGNORE INTO predictions (id, img_id, iou, conf, cat, supercat) VALUES (?, ?, ?, ?, ?, ?)",
                            all_row,
                        )
                else:
                    row = []
                    for id, (img_id, vec, label) in enumerate(
                        zip(img_ids, emb.cpu(), labels["cls"])
                    ):
                        cpu_labels = label.numpy()
                        values, counts = np.unique(cpu_labels, return_counts=True)
                        most_frequent_cat = values[np.argmax(counts)]
                        super_cat = [cat_to_super[cat] for cat in cpu_labels]
                        values, counts = np.unique(super_cat, return_counts=True)
                        most_frequent_supercat = values[np.argmax(counts)]
                        row.append([
                            int((batch_ids * batch_size) + id),
                            int(img_id),
                            vec.numpy().tobytes(),
                            0,
                            0,
                            0,
                            most_frequent_cat,
                            most_frequent_supercat,
                        ])
                    emb_cursor.executemany(
                        "INSERT OR IGNORE INTO embeddings (id, img_id, embedding, hit_freq, mean_iou, mean_conf, flag_cat, flag_supercat) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        row,
                    )

                # Save GAP and bottleneck features
                if "yolo" in self.model_name and 'result' in locals():
                    features_rows = []
                    bottleneck_features = result.get('bottleneck_features', {})
                    gap_features_local = result.get('gap_features', {})
                    structured_preds_local = result.get('predictions', [])
                    for id, img_id in enumerate(img_ids):
                        gap_feat_dict = {k: v[id].cpu().numpy() if isinstance(v, torch.Tensor) else v[id] 
                                        for k, v in gap_features_local.items()}
                        gap_feat_blob = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                   for k, v in gap_feat_dict.items()})

                        bottleneck_blob = None
                        if bottleneck_features:
                            bottleneck_dict = {k: v[id].cpu().numpy() if isinstance(v, torch.Tensor) else v[id] 
                                              for k, v in bottleneck_features.items()}
                            bottleneck_blob = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                         for k, v in bottleneck_dict.items()})

                        features_rows.append((
                            int(img_id),
                            gap_feat_blob,
                            bottleneck_blob,
                            int(time.time()),
                        ))
                    if features_rows:
                        emb_cursor.executemany(
                            "INSERT OR REPLACE INTO features (img_id, gap_features, bottleneck_features, created_ts) VALUES (?, ?, ?, ?)",
                            features_rows,
                        )

                    # Save individual predictions (optional)
                    if self.store_individual_predictions and structured_preds_local is not None:
                        pred_rows = []
                        for id, img_id in enumerate(img_ids):
                            if id < len(structured_preds_local):
                                pred = structured_preds_local[id]
                            else:
                                if self.debug:
                                    self.logger.warning(f"Batch {batch_ids}: missing prediction for image index {id} (img_id={img_id}); using empty prediction")
                                pred = {
                                    'boxes': torch.zeros((0, 4)),
                                    'scores': torch.zeros((0,)),
                                    'classes': torch.zeros((0,)),
                                    'masks': None,
                                }

                            boxes = pred['boxes']
                            scores = pred['scores']
                            classes = pred['classes']
                            masks = pred.get('masks')

                            for box_idx in range(len(boxes)):
                                box = boxes[box_idx]
                                x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                                conf = scores[box_idx].item()
                                cls_id = int(classes[box_idx].item())
                                cls_name = f"class_{cls_id}"

                                mask_blob = None
                                if masks is not None and box_idx < len(masks):
                                    mask_array = masks[box_idx].cpu().numpy().astype(np.float32)
                                    mask_blob = array_to_blob(mask_array)

                                pred_rows.append((
                                    int(img_id),
                                    float(x1),
                                    float(y1),
                                    float(x2),
                                    float(y2),
                                    float(conf),
                                    cls_id,
                                    cls_name,
                                    mask_blob,
                                    int(time.time()),
                                ))
                        if pred_rows:
                            emb_cursor.executemany(
                                "INSERT INTO predictions_individual (img_id, box_x1, box_y1, box_x2, box_y2, confidence, class_id, class_name, mask, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                pred_rows,
                            )

                emb_conn.commit()
                idx += len(images)
                new_embeddings += len(images) 

        # Recompute actual embedding count to avoid counting images that were skipped by INSERT OR IGNORE
        emb_cursor.execute('SELECT COUNT(*) FROM embeddings')
        final_idx = int(emb_cursor.fetchone()[0])

        added = final_idx - metadata_count
        emb_cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', ("total_count", str(final_idx)))
        emb_conn.commit()
        if added > 0:
            self.logger.info(f"Embedding computation complete. Added {added} new embeddings. Total: {final_idx}")
        else:
            self.logger.info(f"All {final_idx} embeddings already computed. Skipping embedding computation step.")
        # use canonical final count for downstream steps
        idx = final_idx

        # Build vector index now that embeddings table is up-to-date
        build_vector_index(emb_conn)
        emb_conn.close()
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        emb_cursor = emb_conn.cursor()

        # For YOLO models we compute head tensors on-the-fly for distance computation
        # and do not store raw predictions in the embeddings DB by default (to avoid large storage).
        if "yolo" in self.model_name:
            if self.store_individual_predictions:
                self.logger.info("YOLO model: storing per-detection predictions in DB (enabled)")
            else:
                self.logger.info("YOLO model: per-detection predictions storage is disabled (masks/bboxes will not be saved)")

        # Step 2: Compute distances with efficient resume
        n_samples = len(data_loader.dataset)
        # Ensure we cover the final partial block even if the dataloader was created with drop_last=True
        n_blocks = (n_samples + batch_size - 1) // batch_size

        # Infer resume blocks (progress table takes precedence)
        start_i_block, start_j_block = (0, 0)
        try:
            start_i_block, start_j_block = self._infer_resume_blocks(dist_cursor, emb_cursor, batch_size, n_blocks, n_samples)
        except sqlite3.OperationalError as e:
            # Missing 'progress' table or other DB error — start from the beginning
            self.logger.warning(f"Could not infer resume blocks (progress table missing or DB error): {e}; starting from i_block=0,j_block=0")
            start_i_block, start_j_block = 0, 0
        except Exception as e:
            # Unexpected error — re-raise to show full traceback
            self.logger.error(f"Unexpected error while inferring resume blocks: {e}")
            raise
        
        if "yolo" in self.model_name:
            if eval_mode:
                self.logger.info(f"Distance computation method: YOLO evaluation-mode metrics (symmetric)")
            else:
                self.logger.info(f"Distance computation method: YOLO training-mode loss (asymmetric)")
        else:
            self.logger.info(f"Distance computation method: Embedding L2 distance")

        def load_block(i_block):
            start = int(i_block * batch_size)
            end = int(min((i_block + 1) * batch_size, n_samples))
            emb_cursor.execute(
                "SELECT img_id, embedding FROM embeddings WHERE id >= ? AND id < ?",
                (start, end),
            )
            idx, emb = [], []
            for eid, blob in emb_cursor.fetchall():
                idx.append(int(eid))
                emb.append(np.frombuffer(blob, dtype=np.float32))
            return idx, torch.from_numpy(np.stack(emb)).to(device)

        for i_block in range(start_i_block, n_blocks):
            idx_i, emb_i = load_block(i_block)

            if emb_i is None or emb_i.shape[0] == 0:
                self.logger.info(f"Skipping empty line {i_block}: {emb_i.cpu()}")
                continue
            # For YOLO models, compute distances strictly from stored raw heads (class+seg loss).
            if "yolo" in self.model_name:
                def load_predictions_for_block(block_idx):
                    """Compute predictions for the block (expensive operation).
                    
                    Returns: (ids, id_to_preds) where id_to_preds maps img_id -> prediction dict
                    """
                    start = int(block_idx * batch_size)
                    end = int(min((block_idx + 1) * batch_size, n_samples))
                    emb_cursor.execute(
                        "SELECT id, img_id FROM embeddings WHERE id >= ? AND id < ?",
                        (start, end),
                    )
                    rows_local = emb_cursor.fetchall()
                    data = [data_loader.dataset.__getitem__(rid[0]) for rid in rows_local]
                    ids, images, _ = data_loader.dataset.collate_fn(data)
                    images = images.to(device)
                    
                    # run model in eval mode to extract final predictions (NMS applied inside wrapper)
                    if hasattr(self.model, 'model') and getattr(self.model, 'model', None) is not None:
                        try:
                            self.model.model.eval()
                        except Exception:
                            # Defensive: some dummy test harnesses set model to None
                            pass
                    # run predictor to get predictions
                    result = self.model.run_with_predictor(images, conf=0.25, iou=0.45)
                    structured_preds = result['predictions']
                    
                    # Build mapping from image_id to predictions
                    id_to_preds = {}
                    for i in range(len(ids)):
                        preds_i = structured_preds[i] if i < len(structured_preds) else {'boxes': torch.tensor([]), 'classes': torch.tensor([])}
                        id_to_preds[int(ids[i])] = preds_i
                    
                    return ids, id_to_preds

                def load_images_for_block(block_idx):
                    """Load images for the block (no inference).
                    
                    Returns: (ids, id_to_images) where id_to_images maps img_id -> image tensor on device
                    """
                    start = int(block_idx * batch_size)
                    end = int(min((block_idx + 1) * batch_size, n_samples))
                    emb_cursor.execute(
                        "SELECT id, img_id FROM embeddings WHERE id >= ? AND id < ?",
                        (start, end),
                    )
                    rows_local = emb_cursor.fetchall()
                    data = [data_loader.dataset.__getitem__(rid[0]) for rid in rows_local]
                    ids, images, _ = data_loader.dataset.collate_fn(data)
                    images = images.to(device)
                    
                    # Build mapping from image_id to image tensors
                    id_to_images = {}
                    for i in range(len(ids)):
                        id_to_images[int(ids[i])] = images[i].detach()
                    
                    return ids, id_to_images

                # Compute i-block predictions ONCE before the j-loop (not N times)
                self.logger.info(f"Computing YOLO predictions for i_block {i_block}/{n_blocks}...")
                idx_i_heads, id2preds_i = load_predictions_for_block(i_block)
                if len(idx_i_heads) == 0:
                    self.logger.debug(f"Empty head block at i_block={i_block}")
                    continue
                self.logger.info(f"Computed predictions for i_block={i_block}: {len(idx_i_heads)} images with detections")

                # Accumulator for batching inserts
                accumulated_rows = []
                COMMIT_EVERY_N_BLOCKS = 1  # Batch commits every 50 block pairs
                sentinel_loss = 424242.0

                j_start = start_j_block if i_block == start_i_block else 0
                for j_block in range(j_start, n_blocks):
                    if j_block % 10 == 0 or j_block == n_blocks - 1:
                        self.logger.info(f"  Computing {'evaluation' if eval_mode else 'losses'}: i_block={i_block}, j_block={j_block}/{n_blocks}")

                    if eval_mode:
                        # Vectorized evaluation: process i and j lists in sub-chunks and compute all-pairs metrics with batched GPU ops when requested
                        idx_j_heads, id2preds_j = load_predictions_for_block(j_block)
                        if len(idx_j_heads) == 0:
                            raise RuntimeError(f"Empty head block at j_block={j_block}")

                        preds_i_list = [id2preds_i[i_img_id] for i_img_id in idx_i_heads]
                        preds_j_list = [id2preds_j[j_img_id] for j_img_id in idx_j_heads]

                        # Estimate safe flatten size using GPU memory (reuse safety factor logic)
                        safety_factor = getattr(self, 'flatten_safety_factor', 0.6)
                        max_flatten_images_env = int(os.environ.get('MAX_FLATTEN_IMAGES', '50000'))

                        # Try to estimate bytes per mask using first available mask shape from preds_j_list
                        mask_h = mask_w = None
                        for p in preds_j_list:
                            m = p.get('masks', None)
                            if m is not None:
                                if torch.is_tensor(m) and m.numel() > 0:
                                    mask_h, mask_w = m.shape[-2], m.shape[-1]
                                    break
                                elif isinstance(m, (list, tuple)) and len(m) > 0:
                                    mm = m[0]
                                    try:
                                        mask_h, mask_w = mm.shape[-2], mm.shape[-1]
                                        break
                                    except Exception:
                                        continue

                        # bytes per mask (bool) roughly 1 byte; per box negligible compared to masks
                        bytes_per_mask = (mask_h * mask_w) if mask_h and mask_w else 1
                        try:
                            total_mem = torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0
                            usable_mem = int(total_mem * float(safety_factor)) if total_mem else 32 * 1024 ** 3
                            max_flatten_images = max(1, min(max_flatten_images_env, int(usable_mem // max(1, bytes_per_mask))))
                        except Exception as e:
                            self.logger.warning(f"Could not determine GPU memory properties for eval chunking: {e}")
                            max_flatten_images = 4096

                        num_i = len(preds_i_list)
                        n_j = len(preds_j_list)
                        # choose i_chunk_size conservatively (reuse previous heuristic)
                        i_chunk_size = max(1, min(num_i, max(1, max_flatten_images // max(1, n_j))))

                        # j_chunk is computed so that i_chunk * j_chunk <= max_flatten_images
                        # We'll dynamically compute j_chunk inside the i_chunk loop based on i_chunk_size

                        use_gpu = bool(eval_on_gpu and torch.cuda.is_available() and str(device).lower().startswith("cuda"))
                        if use_gpu:
                            self.logger.info("Running evaluation metrics on GPU (eval_on_gpu=True)")
                        else:
                            if eval_on_gpu:
                                self.logger.warning("eval_on_gpu requested but CUDA is not available or device is not CUDA — falling back to CPU")

                        # Process chunks
                        for i_start in range(0, num_i, i_chunk_size):
                            i_end = min(i_start + i_chunk_size, num_i)
                            i_chunk_preds = preds_i_list[i_start:i_end]
                            i_chunk_ids = idx_i_heads[i_start:i_end]
                            # determine j_chunk_size based on current i_chunk
                            j_chunk_size = max(1, min(n_j, max(1, max_flatten_images // max(1, i_end - i_start))))

                            for j_start in range(0, n_j, j_chunk_size):
                                j_end = min(j_start + j_chunk_size, n_j)
                                j_chunk_preds = preds_j_list[j_start:j_end]
                                j_chunk_ids = idx_j_heads[j_start:j_end]

                                target_dev = device if use_gpu else torch.device('cpu')

                                try:
                                    # Build padded boxes tensors
                                    def _pad_boxes_list(preds_chunk, target_device):
                                        boxes_list = [p.get('boxes', torch.tensor([])) for p in preds_chunk]
                                        counts = [b.shape[0] if torch.is_tensor(b) else 0 for b in boxes_list]
                                        maxb = max(counts) if len(counts) > 0 else 0
                                        if maxb == 0:
                                            return torch.zeros((len(boxes_list), 0, 4), device=target_device), torch.tensor(counts)
                                        padded = torch.zeros((len(boxes_list), maxb, 4), device=target_device)
                                        for idx_b, b in enumerate(boxes_list):
                                            if torch.is_tensor(b) and b.numel() > 0:
                                                padded[idx_b, : b.shape[0], :] = b.detach().to(target_device).float()
                                        return padded, torch.tensor(counts)

                                    boxes_i_pad, counts_i = _pad_boxes_list(i_chunk_preds, target_dev)
                                    boxes_j_pad, counts_j = _pad_boxes_list(j_chunk_preds, target_dev)

                                    # Vectorized box IoU computation
                                    # boxes_i_pad: [ni, bi,4], boxes_j_pad: [nj, bj,4]
                                    ni, bi, _ = boxes_i_pad.shape
                                    nj, bj, _ = boxes_j_pad.shape

                                    if bi == 0 and bj == 0:
                                        box_sim_mat = torch.ones((ni, nj), device=target_dev)
                                    elif bi == 0 or bj == 0:
                                        box_sim_mat = torch.zeros((ni, nj), device=target_dev)
                                    else:
                                        a = boxes_i_pad
                                        b = boxes_j_pad
                                        a_x1 = a[:,:,0].unsqueeze(2).expand(ni, bi, bj)
                                        a_y1 = a[:,:,1].unsqueeze(2).expand(ni, bi, bj)
                                        a_x2 = a[:,:,2].unsqueeze(2).expand(ni, bi, bj)
                                        a_y2 = a[:,:,3].unsqueeze(2).expand(ni, bi, bj)
                                        b_x1 = b[:,:,0].unsqueeze(1).expand(ni, bi, bj)
                                        b_y1 = b[:,:,1].unsqueeze(1).expand(ni, bi, bj)
                                        b_x2 = b[:,:,2].unsqueeze(1).expand(ni, bi, bj)
                                        b_y2 = b[:,:,3].unsqueeze(1).expand(ni, bi, bj)
                                        ix1 = torch.max(a_x1, b_x1)
                                        iy1 = torch.max(a_y1, b_y1)
                                        ix2 = torch.min(a_x2, b_x2)
                                        iy2 = torch.min(a_y2, b_y2)
                                        iw = (ix2 - ix1).clamp(min=0)
                                        ih = (iy2 - iy1).clamp(min=0)
                                        inter = iw * ih
                                        area_a = ((a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])).unsqueeze(2).expand(ni, bi, bj)
                                        area_b = ((b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])).unsqueeze(1).expand(ni, bi, bj)
                                        union = area_a + area_b - inter
                                        iou = inter / (union + 1e-6)
                                        a_to_b = iou.max(dim=1).values.mean(dim=1)  # [ni]
                                        b_to_a = iou.max(dim=2).values.mean(dim=0)  # [bj]
                                        # create full matrix of mean matches per pair
                                        # compute per-pair a->b and b->a means
                                        a_to_b_mat = iou.max(dim=2).values.mean(dim=1, keepdim=False)  # [ni]
                                        # a_to_b_mat currently [ni], but need per pair directional: instead, compute per pair means
                                        a_to_b_per_pair = iou.max(dim=2).values.mean(dim=1)  # [ni]
                                        # For symmetry, compute pairwise as mean of row-wise and col-wise max averages
                                        # Simpler scalar per pair: compute max per box then average per row and per column
                                        row_max = iou.max(dim=2).values  # [ni, bi]
                                        row_mean = row_max.mean(dim=1, keepdim=True)  # [ni,1]
                                        col_max = iou.max(dim=1).values  # [ni, bj]
                                        col_mean = col_max.mean(dim=0, keepdim=True)  # [1,bj]
                                        # pairwise sim approximated by (row_mean + col_mean)/2 broadcasted
                                        box_sim_mat = (row_mean.expand(ni, nj) + col_mean.expand(ni, nj)) / 2.0

                                    # Segmentation masks vectorized (per-class)
                                    if self.is_seg:
                                        # Determine class union for this chunk pair
                                        # Initialize seg_mean_mat with safe default (ones), will be overwritten if classes exist
                                        seg_mean_mat = torch.ones((ni, nj), device=target_dev) if self.is_seg else None
                                        classes_set = set()
                                        for p in i_chunk_preds:
                                            cls = p.get('classes', torch.tensor([], dtype=torch.long))
                                            if torch.is_tensor(cls):
                                                classes_set.update([int(x) for x in cls.tolist()])
                                            else:
                                                classes_set.update(list(cls))
                                        for p in j_chunk_preds:
                                            cls = p.get('classes', torch.tensor([], dtype=torch.long))
                                            if torch.is_tensor(cls):
                                                classes_set.update([int(x) for x in cls.tolist()])
                                            else:
                                                classes_set.update(list(cls))
                                        classes_list = sorted(list(classes_set))

                                        if len(classes_list) == 0:
                                            seg_mean_mat = torch.ones((ni, nj), device=target_dev)
                                        else:
                                            C = len(classes_list)
                                            # build per-image per-class masks
                                            def _build_masks_tensor(preds_chunk, classes_list, target_device):
                                                # returns [n, C, H, W] bool tensor
                                                # find mask H,W
                                                H = W = None
                                                for p in preds_chunk:
                                                    m = p.get('masks', None)
                                                    if m is not None and getattr(m, 'numel', lambda:0)() > 0:
                                                        m0 = m[0] if not torch.is_tensor(m) else m
                                                        H, W = m0.shape[-2], m0.shape[-1]
                                                        break
                                                if H is None or W is None:
                                                    # no masks in this chunk
                                                    return torch.zeros((len(preds_chunk), C, 1, 1), dtype=torch.bool, device=target_device)
                                                masks_tensor = torch.zeros((len(preds_chunk), C, H, W), dtype=torch.bool, device=target_device)
                                                for ii_p, p in enumerate(preds_chunk):
                                                    m = p.get('masks', None)
                                                    cls = p.get('classes', torch.tensor([], dtype=torch.long))
                                                    if m is None or getattr(m, 'numel', lambda:0)() == 0:
                                                        continue
                                                    if not torch.is_tensor(m):
                                                        m = torch.tensor(m, dtype=torch.float32, device=target_device)
                                                    else:
                                                        m = m.to(target_device)
                                                    if torch.is_tensor(cls):
                                                        cls_list = [int(x) for x in cls.tolist()]
                                                    else:
                                                        cls_list = list(cls)
                                                    for k_c, cid in enumerate(classes_list):
                                                        # find indices where cls_list == cid
                                                        for idx_mask, cid_mask in enumerate(cls_list):
                                                            if cid_mask == cid:
                                                                mask_k = m[idx_mask] > 0.5
                                                                masks_tensor[ii_p, k_c] = masks_tensor[ii_p, k_c] | mask_k
                                                return masks_tensor

                                            masks_i_t = _build_masks_tensor(i_chunk_preds, classes_list, target_dev)
                                            masks_j_t = _build_masks_tensor(j_chunk_preds, classes_list, target_dev)
                                            # Compute per-pair per-class Dice in a memory-safe nested loop
                                            seg_mean_mat = torch.zeros((ni, nj), device=target_dev, dtype=torch.float32)
                                            if C == 0:
                                                seg_mean_mat[:] = 1.0
                                            else:
                                                for ai in range(ni):
                                                    for bj in range(nj):
                                                        dices = []
                                                        for c_idx in range(C):
                                                            a_mask = masks_i_t[ai, c_idx]
                                                            b_mask = masks_j_t[bj, c_idx]
                                                            # both empty masks -> treat as perfect overlap
                                                            if a_mask.numel() == 0 and b_mask.numel() == 0:
                                                                dices.append(1.0)
                                                                continue
                                                            if a_mask.shape != b_mask.shape:
                                                                dices.append(0.0)
                                                                continue
                                                            a_f = a_mask.float()
                                                            b_f = b_mask.float()
                                                            inter = (a_f * b_f).sum()
                                                            sum_a = a_f.sum()
                                                            sum_b = b_f.sum()
                                                            if float(sum_a) == 0.0 and float(sum_b) == 0.0:
                                                                dices.append(1.0)
                                                            else:
                                                                dice = (2.0 * inter) / (sum_a + sum_b + 1e-6)
                                                                dices.append(float(dice))
                                                        seg_mean_mat[ai, bj] = float(sum(dices) / max(1, len(dices)))

                                    # Now assemble rows for all pairs
                                    for a_idx, iid in enumerate(i_chunk_ids):
                                        for b_idx, jid in enumerate(j_chunk_ids):
                                            box_dist = 1.0 - float(box_sim_mat[a_idx, b_idx].item()) if (bi != 0 or bj != 0) else 0.0
                                            cls_jacc = _class_jaccard(i_chunk_preds[a_idx].get('classes', torch.tensor([], dtype=torch.long)), j_chunk_preds[b_idx].get('classes', torch.tensor([], dtype=torch.long)))
                                            cls_dist = 1.0 - cls_jacc
                                            if self.is_seg:
                                                # seg_mean_mat may not be defined in some execution paths; fall back to perfect overlap
                                                try:
                                                    seg_val = float(seg_mean_mat[a_idx, b_idx].item())
                                                except Exception:
                                                    seg_val = 1.0
                                                seg_dist = 1.0 - seg_val
                                            else:
                                                seg_dist = sentinel_loss
                                            dfl_dist = sentinel_loss
                                            accumulated_rows.append((int(iid), int(jid), 'box', float(box_dist)))
                                            accumulated_rows.append((int(iid), int(jid), 'cls', float(cls_dist)))
                                            accumulated_rows.append((int(iid), int(jid), 'dfl', float(dfl_dist)))
                                            if self.is_seg:
                                                accumulated_rows.append((int(iid), int(jid), 'seg', float(seg_dist)))

                                except RuntimeError as e:
                                    # Fallback: on OOM or unexpected runtime issues, fall back to per-pair CPU computation
                                    if 'out of memory' in str(e).lower():
                                        self.logger.warning(f"CUDA OOM during vectorized eval chunk (i:{i_start}-{i_end}, j:{j_start}-{j_end}); falling back to CPU per-pair computation for this chunk")
                                        torch.cuda.empty_cache()
                                    else:
                                        self.logger.exception(f"Error during vectorized eval chunk: {e}; falling back to CPU per-pair computation for this chunk")
                                    # CPU per-pair fallback
                                    for a_idx, iid in enumerate(i_chunk_ids):
                                        pred_i = i_chunk_preds[a_idx]
                                        masks_i_cpu = None
                                        if self.is_seg:
                                            masks_i_cpu = _agg_class_masks(pred_i, torch.device('cpu'))
                                        classes_i = pred_i.get('classes', torch.tensor([], dtype=torch.long))
                                        boxes_i = pred_i.get('boxes', torch.tensor([]))
                                        for b_idx, jid in enumerate(j_chunk_ids):
                                            pred_j = j_chunk_preds[b_idx]
                                            masks_j_cpu = None
                                            if self.is_seg:
                                                masks_j_cpu = _agg_class_masks(pred_j, torch.device('cpu'))
                                            classes_j = pred_j.get('classes', torch.tensor([], dtype=torch.long))
                                            boxes_j = pred_j.get('boxes', torch.tensor([]))

                                            cls_jacc = _class_jaccard(classes_i, classes_j)
                                            cls_dist = 1.0 - cls_jacc
                                            box_sim = _box_similarity(boxes_i, boxes_j, torch.device('cpu'))
                                            box_dist = 1.0 - box_sim
                                            if self.is_seg:
                                                all_classes = set(list(masks_i_cpu.keys()) + list(masks_j_cpu.keys()))
                                                if len(all_classes) == 0:
                                                    seg_mean_dice = 1.0
                                                else:
                                                    dices = []
                                                    for cid in all_classes:
                                                        a = masks_i_cpu.get(cid, torch.zeros_like(next(iter(masks_j_cpu.values())) if masks_j_cpu else torch.zeros((1,1), dtype=torch.bool)))
                                                        b = masks_j_cpu.get(cid, torch.zeros_like(a))
                                                        if a.shape != b.shape:
                                                            dices.append(0.0)
                                                        else:
                                                            dices.append(_dice_mask(a, b))
                                                    seg_mean_dice = float(sum(dices) / max(1, len(dices)))
                                                seg_dist = 1.0 - seg_mean_dice
                                            else:
                                                seg_dist = sentinel_loss

                                            dfl_dist = sentinel_loss
                                            accumulated_rows.append((int(iid), int(jid), 'box', float(box_dist)))
                                            accumulated_rows.append((int(iid), int(jid), 'cls', float(cls_dist)))
                                            accumulated_rows.append((int(iid), int(jid), 'dfl', float(dfl_dist)))
                                            if self.is_seg:
                                                accumulated_rows.append((int(iid), int(jid), 'seg', float(seg_dist)))

                    else:
                        # Training-mode loss path (unchanged)
                        idx_j_heads, id2images_j = load_images_for_block(j_block)
                        if len(idx_j_heads) == 0:
                            raise RuntimeError(f"Empty image block at j_block={j_block}")

                        # Stack all j images into a single batch for efficient processing
                        images_j_batch = torch.stack([id2images_j[img_id] for img_id in idx_j_heads], dim=0)  # [n_j, C, H, W]
                        n_j = len(idx_j_heads)
                        sentinel_loss = 424242.0  # Sentinel value to flag missing/undefined losses
                        
                        # Set model to training mode for loss computation
                        self.model.model.model.train()
                        
                        # Flattened-chunk computation: compute losses for many (i,j) pairs in few forwards
                        preds_list = [id2preds_i[i_img_id] for i_img_id in idx_i_heads]
                        num_i = len(preds_list)

                        # Determine safe flatten size based on GPU memory and image size
                        # safety factor leaves room for activations and other allocations
                        safety_factor = getattr(self, 'flatten_safety_factor', 0.6)
                        max_flatten_images_env = int(os.environ.get('MAX_FLATTEN_IMAGES', '50000'))

                        # bytes per image (use first j image as representative)
                        bytes_per_image = images_j_batch[0].numel() * images_j_batch[0].element_size()
                        # total GPU memory (bytes)
                        try:
                            total_mem = torch.cuda.get_device_properties(images_j_batch.device).total_memory
                            usable_mem = int(total_mem * float(safety_factor))
                            max_flatten_images = max(1, min(max_flatten_images_env, int(usable_mem // max(1, bytes_per_image))))
                        except Exception as e:
                            self.logger.warning(f"Could not determine GPU memory properties: {e}")
                            max_flatten_images = 32

                        # choose chunk size m such that m * n_j <= max_flatten_images
                        i_chunk_size = max(1, min(num_i, max(1, max_flatten_images // max(1, n_j))))

                        # precompute number of predictions per i
                        n_preds_list = [len(p.get('boxes', [])) if isinstance(p, dict) else 0 for p in preds_list]

                        for start in range(0, num_i, i_chunk_size):
                            end = min(start + i_chunk_size, num_i)
                            chunk_preds = preds_list[start:end]
                            chunk_ids = idx_i_heads[start:end]
                            m = len(chunk_preds)

                            # Expand images: ordering ensures losses map to (i_local, j_idx)
                            images_expanded = images_j_batch.unsqueeze(0).repeat(m, 1, 1, 1, 1).reshape(-1, images_j_batch.shape[1], images_j_batch.shape[2], images_j_batch.shape[3])

                            # Build flattened targets by concatenating per-i replicated targets
                            bboxes_parts, cls_parts, batch_idx_parts, masks_parts = [], [], [], []
                            for i_local, pred in enumerate(chunk_preds):
                                if isinstance(pred, dict) and 'boxes' in pred:
                                    boxes = pred['boxes']
                                    classes = pred['classes']
                                    masks = pred.get('masks', None)
                                    n_preds = len(boxes)
                                    if n_preds > 0:
                                        bb = boxes.to(device).float().unsqueeze(0).expand(n_j, -1, -1).reshape(-1, 4)
                                        cc = classes.to(device).long().unsqueeze(0).expand(n_j, -1).reshape(-1)
                                        idxs = (torch.arange(n_j, device=device, dtype=torch.long) + (i_local * n_j)).unsqueeze(1).expand(-1, n_preds).reshape(-1)
                                        bboxes_parts.append(bb)
                                        cls_parts.append(cc)
                                        batch_idx_parts.append(idxs)
                                        if self.is_seg and masks is not None:
                                            if torch.is_tensor(masks):
                                                ms = masks.to(device)
                                            else:
                                                ms = torch.tensor(masks, device=device, dtype=torch.float32)
                                            ms_rep = ms.unsqueeze(0).expand(n_j, -1, -1, -1).reshape(-1, ms.shape[-2], ms.shape[-1])
                                            masks_parts.append(ms_rep)
                                else:
                                    # background: contribute no targets
                                    continue

                            if bboxes_parts:
                                batch_bboxes = torch.cat(bboxes_parts, dim=0)
                                batch_cls = torch.cat(cls_parts, dim=0)
                                batch_idx = torch.cat(batch_idx_parts, dim=0)
                            else:
                                batch_bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                                batch_cls = torch.zeros((0,), device=device, dtype=torch.long)
                                batch_idx = torch.zeros((0,), device=device, dtype=torch.long)

                            batch_targets = {
                                "bboxes": batch_bboxes,
                                "cls": batch_cls,
                                "batch_idx": batch_idx,
                                "img": images_expanded,
                            }

                            if self.is_seg and masks_parts:
                                batch_masks = torch.cat(masks_parts, dim=0)
                                batch_targets['masks'] = batch_masks

                            # Single forward for this chunk (let exceptions propagate)
                            loss_output, loss_items = self.model.model.model(batch_targets)

                            # Expect per-sample loss output shape: [m*n_j, comps]
                            if not (isinstance(loss_items, torch.Tensor) and loss_items.dim() == 2):
                                raise RuntimeError("Unexpected loss_items shape from model; expected [B, comps]")

                            comps = loss_items.shape[1]
                            loss_items = loss_items.reshape(m, n_j, comps)

                            for i_local, i_img_id in enumerate(chunk_ids):
                                n_preds = n_preds_list[start + i_local]
                                for j_idx, j_img_id in enumerate(idx_j_heads):
                                    if comps >= 4 and self.is_seg:
                                        box_loss = float(loss_items[i_local, j_idx, 0].item())
                                        seg_loss = float(loss_items[i_local, j_idx, 1].item())
                                        cls_loss = float(loss_items[i_local, j_idx, 2].item())
                                        dfl_loss = float(loss_items[i_local, j_idx, 3].item())
                                    elif comps >= 3:
                                        box_loss = float(loss_items[i_local, j_idx, 0].item())
                                        cls_loss = float(loss_items[i_local, j_idx, 1].item())
                                        dfl_loss = float(loss_items[i_local, j_idx, 2].item())
                                        seg_loss = sentinel_loss
                                    else:
                                        box_loss = cls_loss = dfl_loss = seg_loss = sentinel_loss

                                    # Suspicious-zero handling: mark sentinel when predictions exist but some comps are zero while cls>0
                                    if n_preds > 0 and cls_loss > 0:
                                        if box_loss == 0.0:
                                            box_loss = sentinel_loss
                                        if dfl_loss == 0.0:
                                            dfl_loss = sentinel_loss
                                        if self.is_seg and seg_loss == 0.0:
                                            seg_loss = sentinel_loss

                                    accumulated_rows.append((int(i_img_id), int(j_img_id), 'box', box_loss))
                                    accumulated_rows.append((int(i_img_id), int(j_img_id), 'cls', cls_loss))
                                    accumulated_rows.append((int(i_img_id), int(j_img_id), 'dfl', dfl_loss))
                                    if self.is_seg:
                                        accumulated_rows.append((int(i_img_id), int(j_img_id), 'seg', seg_loss))
                    # Commit accumulated rows every COMMIT_EVERY_N_BLOCKS block pairs
                    if (j_block + 1) % COMMIT_EVERY_N_BLOCKS == 0 or j_block == n_blocks - 1:
                        if accumulated_rows:
                            _exec_with_retry(dist_cursor, "INSERT OR REPLACE INTO distances (i, j, component, distance) VALUES (?, ?, ?, ?)", accumulated_rows)
                            _commit_with_retry(dist_conn)
                            self.logger.debug(f"Committed {len(accumulated_rows)} distance rows for j_blocks up to {j_block}")
                            accumulated_rows = []
                    
                        # Update progress checkpoint after each j_block completes (with retries)
                        _exec_with_retry(dist_cursor, "INSERT OR REPLACE INTO progress (last_i, last_j) VALUES (?, ?)", (int(i_block), int(j_block)))
                        _commit_with_retry(dist_conn)
                
                self.logger.info(f"Completed distance computation for i_block {i_block + 1}/{n_blocks}")
                # continue to next i_block
                continue

            # Compute distances from embeddings (explicit for non-YOLO models). Insert directed 'total' component for each pair.
            for j_block in range(i_block, n_blocks):  # only upper triangle to avoid double work
                idx_j, emb_j = load_block(j_block)
                if emb_j is None:
                    self.logger.info(f"Skipping empty column {j_block}")
                    continue

                dist_block = torch.cdist(emb_i, emb_j, p=2).cpu().numpy()

                rows = []
                for ii, gid_i in enumerate(idx_i):
                    for jj, gid_j in enumerate(idx_j):
                        # upper-triangle ordering: when i_block == j_block, enforce gid_j >= gid_i
                        if i_block == j_block and gid_j < gid_i:
                            continue
                        d = float(dist_block[ii, jj])
                        # insert both directed entries (i->j and j->i) as 'total'
                        rows.append((int(gid_i), int(gid_j), 'total', d))
                        rows.append((int(gid_j), int(gid_i), 'total', d))

                if rows:
                    _exec_with_retry(dist_cursor, "INSERT OR REPLACE INTO distances (i, j, component, distance) VALUES (?, ?, ?, ?)", rows)

            _commit_with_retry(dist_conn)
            self.logger.debug(f"Completed block row {i_block + 1}/{n_blocks}")

        self.logger.info("Distance computation complete. All pairwise distances computed and stored.")
        
        # Manually checkpoint WAL (PASSIVE) to avoid contention; retry on transient locks
        _exec_with_retry(dist_cursor, "PRAGMA wal_checkpoint(PASSIVE)")
        self.logger.info("Completed WAL checkpoint (PASSIVE)")
        
        self.logger.info(f"Results saved to: {embeddings_db} and {distances_db}")
        emb_conn.close()
        dist_conn.close()

        self.logger.info(f"Inserted {idx} embeddings into {embeddings_db} and built index")
        return embeddings_db, distances_db
