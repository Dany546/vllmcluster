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
        # Use a small sample to infer dim
        img_ids, sample_images, labels = next(iter(data_loader))
        sample_images = sample_images.to(device) 
        if "yolo" in self.model_name:
            # Use new predictor-based approach for YOLO models
            targets = self.model.labels_to_targets(labels, sample_images.shape[0]) if labels else None
            result = self.model.run_with_predictor(
                sample_images,
                targets=targets,
                conf=0.25,
                iou=0.45,
                embed_layers=[-2],
            )
            gap_features = result['gap_features']
            gap_list = [gap_features[k] for k in sorted(gap_features.keys())]
            sample_emb = torch.cat(gap_list, dim=1) if gap_list else torch.zeros(sample_images.shape[0], 512, device=device)
            dim = sample_emb.shape[1]
        else:
            # Non-YOLO models (DINO, CLIP)
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

        create_embeddings_table(emb_conn, dim, is_seg=self.is_seg)

        # Ensure 'predictions' table has required columns for compatibility with older DBs
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

        # --- Insert embeddings while regenerating them ---
        emb_conn.close()
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        emb_cursor = emb_conn.cursor()
        total_inserted = 0
        
        for batch_ids, (img_ids, images, labels) in enumerate(data_loader):
            images = images.to(device)
            if "yolo" in self.model_name:
                # Use new predictor-based approach for YOLO models
                # Use pre-loaded transformed images from dataloader (not file paths)
                # because labels are also transformed to match the augmented images
                
                # Convert labels to ultralytics targets dict format for loss computation
                targets = self.model.labels_to_targets(labels, images.shape[0]) if labels else None
                
                result = self.model.run_with_predictor(
                    images,
                    targets=targets,
                    conf=0.25,  # Very low threshold to catch all detections
                    iou=0.45,
                    embed_layers=[-2, -1] if self.debug else [-2],
                )
                
                # Extract structured predictions, losses, and features
                structured_preds = result['predictions']
                losses = result['losses']
                gap_features = result['gap_features']
                
                # Validate that we got the right number of predictions
                current_batch_size = images.shape[0]  # do not overwrite declared `batch_size`
                
                # Log debug info
                if self.debug:
                    n_dets = sum(len(p['boxes']) for p in structured_preds)
                    self.logger.debug(f"Batch {batch_ids}: {n_dets} total detections across {len(structured_preds)} images") 
                    self.logger.debug(f"Batch {batch_ids}: Prediction boxes per image: {[len(p['boxes']) for p in structured_preds]}")
                    if losses:
                        self.logger.debug(f"Losses available: {list(losses[0].keys())}")
                
                # Compute synthetic embeddings from GAP features (concatenate all heads)
                try:
                    gap_list = [gap_features[k] for k in sorted(gap_features.keys())]
                    if gap_list:
                        emb = torch.cat(gap_list, dim=1)
                    else:
                        emb = torch.zeros(len(structured_preds), 512, device=device)
                except Exception as e:
                    self.logger.warning(f"Error computing embeddings from features: {e}")
                    emb = torch.zeros(len(structured_preds), 512, device=device)
                
                # Match predictions to targets and format outputs for DB insertion
                self.model.debug = False  # Pass debug flag to extractor
                outputs, all_output = self.model.process_yolo_batch(
                    structured_preds, targets, labels, losses, emb, cat_to_super=cat_to_super
                )
            else:
                emb = self.model(images)
                outputs = None
                all_output = None

            rows = []
            if "yolo" in self.model_name and outputs is not None:

                if self.is_seg:
                    for id, (img_id, vec, output) in enumerate(zip(img_ids, emb.cpu(), outputs)):
                        miou, mconf, cat, supercat, hit_freq, dice = output[:6]
                        # losses appended at end if compute_losses used; adapt if different
                        # Use defaults if not present
                        box_loss = output[6] if len(output) > 6 else 0.0
                        cls_loss = output[7] if len(output) > 7 else 0.0
                        dfl_loss = output[8] if len(output) > 8 else 0.0
                        seg_loss = output[9] if len(output) > 9 else 0.0
                        emb_bytes = serialize_float32_array(vec.numpy())
                        row = (
                            int((batch_ids * batch_size) + id),
                            int(img_id),
                            emb_bytes,
                            float(hit_freq),
                            float(miou),
                            float(mconf),
                            float(dice),
                            int(cat),
                            int(supercat),
                            float(box_loss),
                            float(cls_loss),
                            float(dfl_loss),
                            float(seg_loss),
                        )
                        rows.append(row)
                else:
                    for id, (img_id, vec, output) in enumerate(zip(img_ids, emb.cpu(), outputs)):
                        miou, mconf, cat, supercat, hit_freq, dice = output[:6]
                        box_loss = output[6] if len(output) > 6 else 0.0
                        cls_loss = output[7] if len(output) > 7 else 0.0
                        dfl_loss = output[8] if len(output) > 8 else 0.0
                        emb_bytes = serialize_float32_array(vec.numpy())
                        row = (
                            int((batch_ids * batch_size) + id),
                            int(img_id),
                            emb_bytes,
                            float(hit_freq),
                            float(miou),
                            float(mconf),
                            int(cat),
                            int(supercat),
                            float(box_loss),
                            float(cls_loss),
                            float(dfl_loss),
                        )
                        rows.append(row)
            else:
                for id, (img_id, vec, label) in enumerate(zip(img_ids, emb.cpu(), labels["cls"])):
                    cpu_labels = label.numpy()
                    values, counts = np.unique(cpu_labels, return_counts=True)
                    most_frequent_cat = int(values[np.argmax(counts)])
                    super_cat = [cat_to_super[cat] for cat in cpu_labels]
                    values2, counts2 = np.unique(super_cat, return_counts=True)
                    most_frequent_supercat = int(values2[np.argmax(counts2)])
                    emb_bytes = serialize_float32_array(vec.numpy())
                    row = (
                        int((batch_ids * batch_size) + id),
                        int(img_id),
                        emb_bytes,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    )
                    rows.append(row)

            if rows:
                insert_embeddings_batch(emb_conn, rows)
                total_inserted += len(rows)

            # Save features and predictions for YOLO models
            if "yolo" in self.model_name:
                # Save GAP and bottleneck features
                features_rows = []
                bottleneck_features = result.get('bottleneck_features', {})
                for id, img_id in enumerate(img_ids):
                    # Save GAP features
                    gap_feat_dict = {k: v[id].cpu().numpy() if isinstance(v, torch.Tensor) else v[id] 
                                    for k, v in gap_features.items()}
                    gap_feat_blob = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v 
                                               for k, v in gap_feat_dict.items()})
                    
                    # Save bottleneck features if available
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
                        int(time.time())
                    ))
                
                if features_rows:
                    emb_cursor.executemany(
                        "INSERT OR REPLACE INTO features (img_id, gap_features, bottleneck_features, created_ts) VALUES (?, ?, ?, ?)",
                        features_rows
                    )
                
                # Save individual predictions (optional, controlled by flag)
                if self.store_individual_predictions:
                    pred_rows = []
                    for id, img_id in enumerate(img_ids):
                        # Guard against mismatched lengths (predictor may sometimes return fewer results)
                        if id < len(structured_preds):
                            pred = structured_preds[id]
                        else:
                            # Fallback empty prediction
                            if self.debug:
                                self.logger.warning(f"Batch {batch_ids}: missing prediction for image index {id} (img_id={img_id}); using empty prediction")
                            pred = {
                                'boxes': torch.zeros((0, 4)),
                                'scores': torch.zeros((0,)),
                                'classes': torch.zeros((0,)),
                                'masks': None,
                            }

                        boxes = pred['boxes']  # [N, 4] xyxy
                        scores = pred['scores']  # [N]
                        classes = pred['classes']  # [N]
                        masks = pred.get('masks')  # [N, H, W] if seg
                        
                        for box_idx in range(len(boxes)):
                            box = boxes[box_idx]
                            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                            conf = scores[box_idx].item()
                            cls_id = int(classes[box_idx].item())
                            cls_name = f"class_{cls_id}"  # Could map to actual class names
                            
                            # Save mask if available
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
                                int(time.time())
                            ))
                    if pred_rows:
                        emb_cursor.executemany(
                            "INSERT INTO predictions_individual (img_id, box_x1, box_y1, box_x2, box_y2, confidence, class_id, class_name, mask, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            pred_rows
                        )
            
            emb_conn.commit()
        # If some samples were skipped because the dataloader dropped the last incomplete batch,
        # process remaining indices explicitly to ensure all images are embedded.
        if total_inserted < len(data_loader.dataset):
            missing = len(data_loader.dataset) - total_inserted
            self.logger.info(f"Detected {missing} missing embeddings; processing remaining samples manually")
            start_idx = total_inserted
            for i in range(start_idx, len(data_loader.dataset), batch_size):
                chunk_idx = list(range(i, min(i + batch_size, len(data_loader.dataset))))
                # fetch dataset items and collate into a batch
                data = [data_loader.dataset.__getitem__(idx) for idx in chunk_idx]
                img_ids, images, labels = data_loader.dataset.collate_fn(data)
                images = images.to(device)

                if "yolo" in self.model_name:
                    targets = self.model.labels_to_targets(labels, images.shape[0]) if labels else None
                    result = self.model.run_with_predictor(
                        images,
                        targets=targets,
                        conf=0.25,
                        iou=0.45,
                        embed_layers=[-2, -1] if self.debug else [-2],
                    )
                    structured_preds = result.get('predictions', [])
                    gap_features = result.get('gap_features', {})
                    try:
                        gap_list = [gap_features[k] for k in sorted(gap_features.keys())]
                        emb = torch.cat(gap_list, dim=1) if gap_list else torch.zeros(len(structured_preds), 512, device=device)
                    except Exception as e:
                        self.logger.warning(f"Error computing embeddings for manual chunk starting at {i}: {e}")
                        emb = torch.zeros(len(structured_preds), 512, device=device)
                    outputs, all_output = self.model.process_yolo_batch(
                        structured_preds, targets, labels, result.get('losses', None), emb, cat_to_super=cat_to_super
                    )
                else:
                    emb = self.model(images)
                    outputs = None
                    all_output = None

                rows = []
                if "yolo" in self.model_name and outputs is not None:
                    if self.is_seg:
                        for id_in_chunk, (img_id, vec, output) in enumerate(zip(img_ids, emb.cpu(), outputs)):
                            miou, mconf, cat, supercat, hit_freq, dice = output[:6]
                            box_loss = output[6] if len(output) > 6 else 0.0
                            cls_loss = output[7] if len(output) > 7 else 0.0
                            dfl_loss = output[8] if len(output) > 8 else 0.0
                            seg_loss = output[9] if len(output) > 9 else 0.0
                            emb_bytes = serialize_float32_array(vec.numpy())
                            row = (
                                int(total_inserted + id_in_chunk),
                                int(img_id),
                                emb_bytes,
                                float(hit_freq),
                                float(miou),
                                float(mconf),
                                float(dice),
                                int(cat),
                                int(supercat),
                                float(box_loss),
                                float(cls_loss),
                                float(dfl_loss),
                                float(seg_loss),
                            )
                            rows.append(row)
                    else:
                        for id_in_chunk, (img_id, vec, output) in enumerate(zip(img_ids, emb.cpu(), outputs)):
                            miou, mconf, cat, supercat, hit_freq, dice = output[:6]
                            box_loss = output[6] if len(output) > 6 else 0.0
                            cls_loss = output[7] if len(output) > 7 else 0.0
                            dfl_loss = output[8] if len(output) > 8 else 0.0
                            emb_bytes = serialize_float32_array(vec.numpy())
                            row = (
                                int(total_inserted + id_in_chunk),
                                int(img_id),
                                emb_bytes,
                                float(hit_freq),
                                float(miou),
                                float(mconf),
                                int(cat),
                                int(supercat),
                                float(box_loss),
                                float(cls_loss),
                                float(dfl_loss),
                            )
                            rows.append(row)
                else:
                    for id_in_chunk, (img_id, vec, label) in enumerate(zip(img_ids, emb.cpu(), labels["cls"])):
                        cpu_labels = label.numpy()
                        values, counts = np.unique(cpu_labels, return_counts=True)
                        most_frequent_cat = int(values[np.argmax(counts)])
                        super_cat = [cat_to_super[cat] for cat in cpu_labels]
                        values2, counts2 = np.unique(super_cat, return_counts=True)
                        most_frequent_supercat = int(values2[np.argmax(counts2)])
                        emb_bytes = serialize_float32_array(vec.numpy())
                        row = (
                            int(total_inserted + id_in_chunk),
                            int(img_id),
                            emb_bytes,
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                        )
                        rows.append(row)
                if rows:
                    insert_embeddings_batch(emb_conn, rows)
                    total_inserted += len(rows)
            emb_conn.commit()

        build_vector_index(emb_conn)
        emb_conn.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ("total_count", str(total_inserted)))
        emb_conn.commit()
        emb_conn.close()

        # Optionally keep distances DB for other uses, but we no longer precompute all pairwise distances here.
        # If you still need a distances DB for compatibility, you can keep creating it but leave it empty or compute on demand.
        dist_conn = sqlite3.connect(distances_db)
        dist_conn.execute("PRAGMA journal_mode=WAL")
        dist_conn.execute("PRAGMA synchronous=OFF")
        dist_conn.execute("PRAGMA temp_store=MEMORY")
        dist_conn.execute("PRAGMA cache_size=-20000")
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
        existing_count = result[0] if result else 0

        if existing_count == 0:
            self.logger.info("Computing embeddings from scratch...")
        else:
            self.logger.info(f"Found {existing_count} existing embeddings ...")

        idx = int(existing_count)
        new_embeddings = 0
        if idx < len(data_loader.dataset):
            data_loader.start_idx = existing_count
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
                emb_conn.commit()
                idx += len(images)
                new_embeddings += len(images)

        if new_embeddings > 0:
            emb_cursor.execute(
                'INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)', ("total_count", str(idx))
            )
            emb_conn.commit()
            self.logger.info(f"Added {new_embeddings} new embeddings. Total: {idx}")
        else:
            self.logger.info(f"All {existing_count} embeddings already computed")

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

        self.logger.info(
            f"Computing distances for {n_samples} samples ({n_blocks} x {n_blocks} blocks)..."
        )

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

        for i_block in range(n_blocks):
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
                def load_head_block_map(block_idx):
                    """Compute head tensors, final predictions, and original images for the block.

                    Returns: (ids, id_to_heads, id_to_preds, id_to_images)
                    """
                    # Compute head tensors for the block directly from the dataset.
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
                    
                    # extract head tensors and store per-image original image tensors
                    B = images.shape[0]  
                    id_to_preds = {}
                    id_to_images = {}
                    for i in range(B):
                        preds_i = structured_preds[i] if i < len(structured_preds) else {'boxes': [], 'classes': []}
                        id_to_preds[int(ids[i])] = preds_i

                        id_to_images[int(ids[i])] = images[i].detach() 

                    return ids, id_to_preds, id_to_images

                # Full directed O(N^2) computation using YOLO training-mode per-image losses.
                # Computes both A->B and B->A for all pairs to build complete asymmetric distance matrix
                def preds_to_targets(preds_tensor: torch.Tensor):
                    """Convert final_predictions tensor (Nx6 or Nx6+mask) to training-style targets dict.
                    
                    Expected columns: x1,y1,x2,y2,conf,cls
                    Returns dict with bboxes, cls, and batch_idx (all boxes treated as batch_idx=0 since single image).
                    """
                    if not isinstance(preds_tensor, torch.Tensor) or preds_tensor.numel() == 0:
                        # Empty targets
                        return {
                            "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                            "cls": torch.zeros((0,), device=device, dtype=torch.int64),
                            "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                        }
                    p = preds_tensor.to(device)
                    # columns: x1,y1,x2,y2,conf,cls,...
                    if p.dim() == 1:
                        p = p.unsqueeze(0)
                    bboxes = p[:, :4].float()
                    cls = p[:, 5].long()
                    n_boxes = bboxes.shape[0]
                    batch_idx = torch.zeros((n_boxes,), device=device, dtype=torch.int64)  # single image = batch_idx 0
                    return {"bboxes": bboxes, "cls": cls, "batch_idx": batch_idx, 'masks': None}

                # preload heads for i_block
                idx_i_heads, id2preds_i, id2images_i = load_head_block_map(i_block)
                if len(idx_i_heads) == 0:
                    self.logger.debug(f"Empty head block at i_block={i_block}")
                    continue

                for j_block in range(0, n_blocks):
                    idx_j_heads, id2preds_j, id2images_j = load_head_block_map(j_block)
                    if len(idx_j_heads) == 0:
                        raise RuntimeError(f"Empty head block at j_block={j_block}")

                    # A -> B (directed): run compute_losses on image A in training mode with B's predictions as targets
                    rows = []
                    self.model.model.model.train()  # ensure training mode for loss computation
                    
                    # For each image in block i
                    for i_idx, i_img_id in enumerate(idx_i_heads):
                        # Get image i's features
                        img_i = id2images_i[i_img_id]
                        
                        # For each image in block j, use its predictions as targets
                        for j_idx, j_img_id in enumerate(idx_j_heads):
                            preds_j = id2preds_j[j_img_id]  # Predictions from image j (dict with boxes, classes, etc)
                            
                            # Convert predictions to targets format
                            if isinstance(preds_j, dict) and 'boxes' in preds_j:
                                # Structured predictions dict
                                boxes = preds_j['boxes']  # [N, 4] xyxy
                                classes = preds_j['classes']  # [N]
                                masks = preds_j.get('masks', None)

                                if len(boxes) == 0:
                                    # No predictions to use as targets
                                    targets_j = {
                                        "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                                        "cls": torch.zeros((0,), device=device, dtype=torch.int64),
                                        "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                                        "masks": None,
                                    }
                                else:
                                    # Convert masks to a device tensor if present, otherwise leave as None
                                    if masks is None:
                                        masks_tensor = None
                                    else:
                                        if torch.is_tensor(masks):
                                            masks_tensor = masks.to(device)
                                        else:
                                            masks_tensor = torch.tensor(masks, device=device, dtype=torch.float32)

                                    targets_j = {
                                        "bboxes": boxes.to(device).float(),
                                        "cls": classes.to(device).long(),
                                        "batch_idx": torch.zeros(len(boxes), device=device, dtype=torch.int64),
                                        "masks": masks_tensor,
                                    }
                            else:
                                # Empty predictions
                                targets_j = {
                                    "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                                    "cls": torch.zeros((0,), device=device, dtype=torch.int64),
                                    "batch_idx": torch.zeros((0,), device=device, dtype=torch.int64),
                                    "masks": None,
                                }
                            
                            # Run forward pass in training mode to compute loss
                            # Loss = how well predictions from j fit image i
                            try:
                                targets_j = {**targets_j, "img": img_i.unsqueeze(0)}  # Add image i
                                loss_output, loss_items = self.model.model.model(targets_j)
                                
                                # Extract per-component losses
                                if isinstance(loss_items, torch.Tensor) and loss_items.numel() >= 3:
                                    box_loss = float(loss_items[0].item()) if loss_items.shape[0] > 0 else 0.0
                                    cls_loss = float(loss_items[1].item()) if loss_items.shape[0] > 1 else 0.0
                                    dfl_loss = float(loss_items[2].item()) if loss_items.shape[0] > 2 else 0.0
                                    seg_loss = float(loss_items[3].item()) if self.is_seg and loss_items.shape[0] > 3 else 0.0
                                else:
                                    box_loss = cls_loss = dfl_loss = seg_loss = 0.0
                                
                                # Store distances for this pair
                                rows.append((int(i_img_id), int(j_img_id), 'box', box_loss))
                                rows.append((int(i_img_id), int(j_img_id), 'cls', cls_loss))
                                rows.append((int(i_img_id), int(j_img_id), 'dfl', dfl_loss))
                                if self.is_seg:
                                    rows.append((int(i_img_id), int(j_img_id), 'seg', seg_loss))
                            except Exception as e:
                                self.logger.warning(f"Error computing loss for pair ({i_img_id}, {j_img_id}): {e}")
                                # Skip this pair
                                continue
                    
                    # Insert all distances for this block pair
                    if rows:
                        dist_cursor.executemany(
                            "INSERT OR REPLACE INTO distances (i, j, component, distance) VALUES (?, ?, ?, ?)",
                            rows,
                        )
                        dist_conn.commit()
                self.logger.debug(f"Completed head-based block row {i_block + 1}/{n_blocks}")
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

        self.logger.info("Distance computation complete")
        emb_conn.close()
        dist_conn.close()

        self.logger.info(f"Inserted {total_inserted} embeddings into {embeddings_db} and built index")
        return embeddings_db, distances_db
