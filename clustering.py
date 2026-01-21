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

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def distance_matrix_db(
        self,
        data_loader,
        pragma_speed: bool = True,
    ) -> Tuple[str, str]:
        """Compute pairwise distances and store embeddings in a vector-enabled SQLite DB."""
        # try to read declared batch size from the dataloader, fallback to inspecting first batch
        batch_size = getattr(data_loader, "batch_size", None)
        if batch_size is None:
            batch_size = len(next(iter(data_loader))[0])
        embeddings_db = self.embeddings_db
        distances_db = self.distances_db  # you can keep this path if you want to store other info

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
        try:
            emb_cursor.execute("PRAGMA table_info(predictions)")
            existing_cols = [r[1] for r in emb_cursor.fetchall()]
            required = ['id', 'img_id', 'iou', 'conf', 'cat', 'supercat']
            if self.is_seg:
                # segmentation models also expect 'dice'
                if 'dice' not in existing_cols:
                    required.insert(4, 'dice')
            for c in required:
                if c not in existing_cols:
                    if c == 'id':
                        self.logger.warning("Existing 'predictions' table missing 'id' column; cannot auto-add primary key")
                    else:
                        try:
                            emb_cursor.execute(f"ALTER TABLE predictions ADD COLUMN {c} BLOB")
                        except Exception as e:
                            self.logger.warning(f"Failed to add column {c} to 'predictions': {e}")
            emb_conn.commit()
        except Exception as e:
            self.logger.debug(f"Could not validate or migrate 'predictions' table: {e}")

# Removed redundant early embedding pass. Embedding extraction and feature/prediction saving
        # are performed in the canonical 'Step 1: Check and compute embeddings' pass below.

        # Optionally keep distances DB for other uses, but we no longer precompute all pairwise distances here.
        # If you still need a distances DB for compatibility, you can keep creating it but leave it empty or compute on demand.
        dist_conn = sqlite3.connect(distances_db)
        dist_conn.execute("PRAGMA journal_mode=WAL")
        dist_conn.execute("PRAGMA synchronous=OFF")
        dist_conn.execute("PRAGMA temp_store=MEMORY")
        dist_conn.execute("PRAGMA cache_size=-1024000")  # 1GB cache for large distance matrix
        dist_conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        dist_conn.execute("PRAGMA locking_mode=EXCLUSIVE")  # Single-writer optimization
        dist_conn.execute("PRAGMA automatic_index=OFF")  # Prevent auto-index during inserts
        dist_conn.execute("PRAGMA wal_autocheckpoint=0")  # Disable auto-checkpoint during writes
        dist_cursor = dist_conn.cursor()
        # Create distances table with a 'component' column (component-aware distances)
        dist_cursor.execute("""
            CREATE TABLE IF NOT EXISTS distances (
                i INTEGER,
                j INTEGER,
                component TEXT,
                distance REAL,
                PRIMARY KEY (i, j, component)
            )
        """)
        dist_cursor.execute(
            "CREATE TABLE IF NOT EXISTS progress (last_i INTEGER PRIMARY KEY, last_j INTEGER)"
        )

        # Create indices for the component-aware distances table
        dist_cursor.execute("CREATE INDEX IF NOT EXISTS idx_i ON distances(i)")
        dist_cursor.execute("CREATE INDEX IF NOT EXISTS idx_j ON distances(j)")
        dist_cursor.execute("CREATE INDEX IF NOT EXISTS idx_component ON distances(component)")
        dist_conn.commit()

        # Step 1: Check and compute embeddings
        # Ensure embeddings DB is open (we closed it earlier after building index)
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        emb_cursor = emb_conn.cursor()
        emb_cursor.execute('SELECT value FROM metadata WHERE key = "total_count"')
        result = emb_cursor.fetchone()
        metadata_count = int(result[0]) if result else 0

        # Verify metadata against actual embeddings table to avoid resuming from stale metadata
        try:
            emb_cursor.execute('SELECT COUNT(*) FROM embeddings')
            actual_count = int(emb_cursor.fetchone()[0])
        except Exception as e:
            self.logger.warning(f"Could not get actual embedding count from DB: {e}")
            actual_count = metadata_count

        if metadata_count != actual_count:
            # Prefer the actual table count (more reliable) and fix metadata
            self.logger.warning(f"Metadata total_count={metadata_count} inconsistent with actual embeddings count={actual_count}. Using actual count and repairing metadata.")
            metadata_count = actual_count
            try:
                emb_cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', ("total_count", str(metadata_count)))
                emb_conn.commit()
            except Exception as e:
                self.logger.warning(f"Failed to repair metadata.total_count: {e}")

        if metadata_count == 0:
            self.logger.info("Computing embeddings from scratch...")
        else:
            self.logger.info(f"Found {metadata_count} existing embeddings ...")

        idx = int(metadata_count)
        new_embeddings = 0

        # Determine embedding dimension `dim` now that we can inspect the embeddings DB
        if idx > 0:
            try:
                emb_cursor.execute("SELECT embedding FROM embeddings LIMIT 1")
                row = emb_cursor.fetchone()
                if row and row[0]:
                    arr = np.frombuffer(row[0], dtype=np.float32)
                    dim = int(arr.shape[0])
                    self.logger.info(f"Inferred embedding dim={dim} from existing embeddings")
                else:
                    dim = 512
                    self.logger.warning("Could not infer dim from DB, defaulting to 512")
            except Exception as e:
                dim = 512
                self.logger.warning(f"Failed to infer dim from DB: {e}. Defaulting to {dim}")
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
                    try:
                        gap_list = [gap_features[k] for k in sorted(gap_features.keys())]
                        if gap_list:
                            emb = torch.cat(gap_list, dim=1)
                        else:
                            emb = torch.zeros(len(structured_preds), 512, device=device)
                    except Exception as e:
                        self.logger.warning(f"Error computing embeddings: {e}")
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
                                miou.tobytes(),
                                mconf.tobytes(),
                                dice.tobytes(),
                                cat.tobytes(),
                                supercat.tobytes(),
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
                                miou.tobytes(),
                                mconf.tobytes(),
                                cat.tobytes(),
                                supercat.tobytes(),
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

        emb_cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', ("total_count", str(idx)))
        emb_conn.commit()
        if new_embeddings > 0:
            self.logger.info(f"Embedding computation complete. Added {new_embeddings} new embeddings. Total: {idx}")
        else:
            self.logger.info(f"All {idx} embeddings already computed. Skipping embedding computation step.")

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

        # Check for existing progress and resume from checkpoint
        dist_cursor.execute("SELECT last_i, last_j FROM progress ORDER BY last_i DESC LIMIT 1")
        progress_row = dist_cursor.fetchone()
        if progress_row:
            resume_i, resume_j = progress_row
            self.logger.info(
                f"Resuming distance computation from checkpoint: i_block={resume_i}, j_block={resume_j+1}"
            )
            start_i_block = resume_i
            start_j_block = resume_j + 1  # Continue from next j_block
            if start_j_block >= n_blocks:
                # Completed all j_blocks for this i, move to next i
                start_i_block += 1
                start_j_block = 0
        else:
            # No explicit progress checkpoint. Check whether the distances table already contains rows
            dist_cursor.execute("SELECT COUNT(*), MAX(i), MAX(j) FROM distances")
            dist_info = dist_cursor.fetchone()
            dist_count = int(dist_info[0]) if dist_info and dist_info[0] is not None else 0
            if dist_count > 0:
                max_i = int(dist_info[1]) if dist_info[1] is not None else 0
                max_j = int(dist_info[2]) if dist_info[2] is not None else 0

                # Infer resume blocks from highest indices observed in distances table
                inferred_i_block = max_i // batch_size
                inferred_j_block = (max_j // batch_size) + 1
                if inferred_j_block >= n_blocks:
                    inferred_i_block += 1
                    inferred_j_block = 0

                self.logger.info(f"Distances DB already contains {dist_count} rows; resuming from inferred position i_block={inferred_i_block}, j_block={inferred_j_block}")
                start_i_block = inferred_i_block
                start_j_block = inferred_j_block
            else:
                self.logger.info(
                    f"Starting distance computation for {n_samples} samples ({n_blocks} x {n_blocks} blocks)..."
                )
                start_i_block = 0
                start_j_block = 0
        
        if "yolo" in self.model_name:
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
            try:
                idx_i, emb_i = load_block(i_block)
            except Exception as e:
                self.logger.error(f"Error loading block {i_block}: {e}")
                raise

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
                    self.model.model.eval()  
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
                COMMIT_EVERY_N_BLOCKS = 50  # Batch commits every 50 block pairs

                j_start = start_j_block if i_block == start_i_block else 0
                for j_block in range(j_start, n_blocks):
                    # Load ONLY images for j_block (no prediction computation)
                    if j_block % 10 == 0 or j_block == n_blocks - 1:
                        self.logger.info(f"  Computing losses: i_block={i_block}, j_block={j_block}/{n_blocks}")
                    idx_j_heads, id2images_j = load_images_for_block(j_block)
                    if len(idx_j_heads) == 0:
                        raise RuntimeError(f"Empty image block at j_block={j_block}")

                    # Set model to training mode for loss computation
                    self.model.model.model.train()
                    
                    # For each prediction in block i, compute loss against each image in block j
                    for i_idx, i_img_id in enumerate(idx_i_heads):
                        preds_i = id2preds_i[i_img_id]  # Predictions from image i (computed once)
                        
                        # Convert predictions to targets format
                        if isinstance(preds_i, dict) and 'boxes' in preds_i:
                            # Structured predictions dict
                            boxes = preds_i['boxes']  # [N, 4] xyxy
                            classes = preds_i['classes']  # [N]
                            masks = preds_i.get('masks', None)

                            if len(boxes) == 0:
                                # No predictions to use as targets
                                targets_i = {
                                    "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                                    "cls": torch.zeros((0,), device=device, dtype=torch.int64),
                                    "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                                    "masks": None,
                                }
                            else:
                                # Convert masks to a device tensor if present
                                if masks is None:
                                    masks_tensor = None
                                else:
                                    if torch.is_tensor(masks):
                                        masks_tensor = masks.to(device)
                                    else:
                                        masks_tensor = torch.tensor(masks, device=device, dtype=torch.float32)

                                targets_i = {
                                    "bboxes": boxes.to(device).float(),
                                    "cls": classes.to(device).long(),
                                    "batch_idx": torch.zeros(len(boxes), device=device, dtype=torch.int64),
                                    "masks": masks_tensor,
                                }
                        else:
                            # Empty predictions
                            targets_i = {
                                "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                                "cls": torch.zeros((0,), device=device, dtype=torch.int64),
                                "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                                "masks": None,
                            }
                        
                        # For each image in j_block, compute how well i's predictions fit j's image
                        for j_idx, j_img_id in enumerate(idx_j_heads):
                            img_j = id2images_j[j_img_id]  # Image from j (no prediction inference needed)
                            
                            # Run forward pass in training mode to compute loss
                            # Loss = how well predictions from i fit image j
                            try:
                                targets_i_with_img = {**targets_i, "img": img_j.unsqueeze(0)}  # Add image j
                                loss_output, loss_items = self.model.model.model(targets_i_with_img)
                                
                                # Extract per-component losses
                                if isinstance(loss_items, torch.Tensor) and loss_items.numel() >= 3:
                                    box_loss = float(loss_items[0].item()) if loss_items.shape[0] > 0 else 0.0
                                    cls_loss = float(loss_items[1].item()) if loss_items.shape[0] > 1 else 0.0
                                    dfl_loss = float(loss_items[2].item()) if loss_items.shape[0] > 2 else 0.0
                                    seg_loss = float(loss_items[3].item()) if self.is_seg and loss_items.shape[0] > 3 else 0.0
                                else:
                                    box_loss = cls_loss = dfl_loss = seg_loss = 0.0
                                
                                # Store distances for this pair
                                accumulated_rows.append((int(i_img_id), int(j_img_id), 'box', box_loss))
                                accumulated_rows.append((int(i_img_id), int(j_img_id), 'cls', cls_loss))
                                accumulated_rows.append((int(i_img_id), int(j_img_id), 'dfl', dfl_loss))
                                if self.is_seg:
                                    accumulated_rows.append((int(i_img_id), int(j_img_id), 'seg', seg_loss))
                            except Exception as e:
                                self.logger.warning(f"Error computing loss for pair ({i_img_id}, {j_img_id}): {e}")
                                # Skip this pair
                                continue
                    
                    # Commit accumulated rows every COMMIT_EVERY_N_BLOCKS block pairs
                    if (j_block + 1) % COMMIT_EVERY_N_BLOCKS == 0 or j_block == n_blocks - 1:
                        if accumulated_rows:
                            dist_cursor.executemany(
                                "INSERT OR REPLACE INTO distances (i, j, component, distance) VALUES (?, ?, ?, ?)",
                                accumulated_rows,
                            )
                            dist_conn.commit()
                            self.logger.debug(f"Committed {len(accumulated_rows)} distance rows for j_blocks up to {j_block}")
                            accumulated_rows = []
                    
                    # Update progress checkpoint after each j_block completes
                    dist_cursor.execute(
                        "INSERT OR REPLACE INTO progress (last_i, last_j) VALUES (?, ?)",
                        (int(i_block), int(j_block))
                    )
                    dist_conn.commit()
                
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
                    dist_cursor.executemany(
                        "INSERT OR REPLACE INTO distances (i, j, component, distance) VALUES (?, ?, ?, ?)",
                        rows,
                    )

            dist_conn.commit()
            self.logger.debug(f"Completed block row {i_block + 1}/{n_blocks}")

        self.logger.info("Distance computation complete. All pairwise distances computed and stored.")
        
        # Manually checkpoint WAL to prevent file bloat with synchronous=OFF
        try:
            dist_conn.execute("PRAGMA wal_checkpoint(RESTART)")
            self.logger.info("Completed WAL checkpoint")
        except Exception as e:
            self.logger.warning(f"WAL checkpoint failed: {e}")
        
        self.logger.info(f"Results saved to: {embeddings_db} and {distances_db}")
        emb_conn.close()
        dist_conn.close()

        self.logger.info(f"Inserted {idx} embeddings into {embeddings_db} and built index")
        return embeddings_db, distances_db
