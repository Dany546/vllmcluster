import logging
import sqlite3
from pydoc import classify_class_attrs
import io
import json
import time

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


class yolowrapper(torch.nn.Module):
    def __init__(self, model_str):
        super().__init__()
        self.model = YOLO(model_str).model 
        args = self.model.args
        args['overlap_mask'] = True
        args['mosaic'] = 0
        args['profile'] = False
        for key in ['box', 'cls', 'dfl']:
            args[key] = 1  
        self.model.args =  SimpleNamespace(**args)  
        self.model = self.model.to(device)
        self.model.device = device
        self.model.criterion = self.model.init_criterion()
        self.model.eval()
        
        # Check if this is a segmentation model
        self.is_seg = 'seg' in model_str.lower()

        def hook_fn(module, input, output):
            module.features = (
                torch.nn.functional.adaptive_avg_pool2d(input[0], (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )

        # inputs of detection heads
        self.detection_layer_entry = self.model.model[-1].cv2
        for m in self.detection_layer_entry:
            m.features = None
            m.register_forward_hook(hook_fn)

    def compute_losses(self, images: torch.Tensor, targets: dict):
        """
        Compute per-image losses using the model's training mode.
        
        Returns:
            per_image_losses: List of dicts with loss components per image
        """
        # Temporarily switch to training mode to compute losses
        was_training = self.model.training
        self.model.train()
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                value.to(device)
        
        try:
            # Forward pass with loss computation
            # Create a copy to avoid issues
            img_copy = images.clone().requires_grad_(False)
            batch = {'img': img_copy, **targets}
            loss_dict = self.model(batch)
            
            # Extract per-image losses from the model's loss output
            # Ultralytics returns (loss, loss_items) where loss_items is a tensor
            if isinstance(loss_dict, tuple):
                try:
                    loss, loss_items = loss_dict
                except Exception as err:
                    print("Error extracting loss items:", len(loss_dict))
                    raise err
                # loss_items typically contains [box_loss, cls_loss, dfl_loss] or 
                # [box_loss, seg_loss, cls_loss, dfl_loss] for segmentation
            else:
                # If it returns dict or single value, handle accordingly
                loss = loss_dict
                loss_items = None
            
            # Get detailed loss breakdown from model's internal loss storage
            # The model stores per-batch losses, we need per-image
            batch_size = images.shape[0]
            
            if self.is_seg:
                # Segmentation model: box_loss, seg_loss, cls_loss, dfl_loss
                if loss_items is not None:
                    box_loss = loss_items[0].item() / batch_size
                    seg_loss = loss_items[1].item() / batch_size if len(loss_items) > 1 else 0.0
                    cls_loss = loss_items[2].item() / batch_size if len(loss_items) > 2 else 0.0
                    dfl_loss = loss_items[3].item() / batch_size if len(loss_items) > 3 else 0.0
                else:
                    box_loss = seg_loss = cls_loss = dfl_loss = 0.0
                
                per_image_losses = [{
                    'box_loss': box_loss,
                    'seg_loss': seg_loss,
                    'cls_loss': cls_loss,
                    'dfl_loss': dfl_loss
                } for _ in range(batch_size)]
            else:
                # Detection model: box_loss, cls_loss, dfl_loss
                if loss_items is not None:
                    box_loss = loss_items[0].item() / batch_size
                    cls_loss = loss_items[1].item() / batch_size if len(loss_items) > 1 else 0.0
                    dfl_loss = loss_items[2].item() / batch_size if len(loss_items) > 2 else 0.0
                else:
                    box_loss = cls_loss = dfl_loss = 0.0
                
                per_image_losses = [{
                    'box_loss': box_loss,
                    'cls_loss': cls_loss,
                    'dfl_loss': dfl_loss
                } for _ in range(batch_size)]
            
        finally:
            # Restore original training state
            if not was_training:
                self.model.eval()
        
        return per_image_losses

    def match_targets_to_preds(
        self, final_preds, batch_idx, targets, N, iou_thres=0.6, conf_thres=0.5
    ):
        """
        final_preds: list of tensors per image, each [N,6] (x1,y1,x2,y2,conf,cls)
                     or [N,6+mask_dim] for segmentation
        batch_idx: indexes of predicted boxes in the batch
        targets: dict with GT boxes and labels (and masks for segmentation)
        returns: filtered targets aligned with final_preds
        """
        matched = []
        all = []
        for img_idx in range(N):
            mask = targets["batch_idx"] == img_idx
            preds = final_preds[batch_idx == img_idx]
            conf = preds[:, 4:5].squeeze(-1)
            preds = preds[conf > conf_thres]
            pred_box = preds[:, :4]
            cls = preds[:, 5]
            conf = preds[:, 4:5]
            
            # Handle masks for segmentation models
            dice_score = 0.0
            if self.is_seg and preds.shape[1] > 6:
                # Extract mask data if available
                pred_masks = preds[:, 6:]  # Remaining columns are mask data
                if 'masks' in targets and mask.sum() > 0:
                    gt_masks = targets['masks'][mask]
                    if len(pred_masks) > 0 and len(gt_masks) > 0:
                        # Compute Dice score
                        dice_score = compute_dice_score(pred_masks, gt_masks).item()
            
            if mask.sum() == 0:
                if len(preds) == 0:
                    miou = 1
                    cat = -1
                    hitfreq = 1
                else:
                    miou = 0
                    cat = -2
                    hitfreq = 0
                matched.append([miou, 1, cat, cat, hitfreq, dice_score])
                all.append(
                    [
                        img_idx,
                        np.array([miou], dtype=np.float32),
                        np.array([1], dtype=np.float32),
                        np.array([cat], dtype=np.float32),
                        np.array([cat], dtype=np.float32),
                        np.array([dice_score], dtype=np.float32),
                    ]
                )
                continue
            gt_boxes = targets["bboxes"][mask].to(device)  # [M,4]
            gt_labels = targets["cls"][mask].to(device).view(-1)  # [M]
            match = []
            all_match = []
            correct = 0
            for idx, g in enumerate(gt_boxes):
                if sum(cls == gt_labels[idx]) > 0:
                    ious = box_iou(pred_box, g.unsqueeze(0))  # [1,M]
                    best_iou, best_idx = ious[cls == gt_labels[idx]].max(0)
                    if best_iou > iou_thres:
                        match += [
                            best_iou,
                            conf[cls == gt_labels[idx]][best_idx],
                            gt_labels[idx],
                            cat_to_super[gt_labels[idx]],
                        ]
                        correct += 1
                        ious = ious[cls == gt_labels[idx]]
                        ious = ious[ious > iou_thres]
                        best_iou, best_idx = ious.sort(0, descending=True)
                        all_match += [
                            img_idx,
                            best_iou,
                            conf[cls == gt_labels[idx]][ious > iou_thres][best_idx],
                            np.ones(len(best_iou)) * gt_labels[idx],
                            np.ones(len(best_iou)) * cat_to_super[gt_labels[idx]],
                            np.ones(len(best_iou)) * dice_score,
                        ]
            correct /= len(gt_boxes)
            cpu_labels = gt_labels.cpu().numpy()
            values, counts = np.unique(cpu_labels, return_counts=True)
            most_frequent_cat = values[np.argmax(counts)]
            super_cat = [cat_to_super[cat] for cat in cpu_labels]
            values, counts = np.unique(super_cat, return_counts=True)
            most_frequent_supercat = values[np.argmax(counts)]
            if len(match) > 0:
                match = np.array(match, dtype=np.float32)
                matched.append(
                    [
                        match[:, 0].mean(),
                        match[:, 1].mean(),
                        most_frequent_cat,
                        most_frequent_supercat,
                        correct,
                        dice_score,
                    ]
                )
                all.append(all_match)
            else:
                matched.append([0, 1, most_frequent_cat, most_frequent_supercat, 0, dice_score])
                all.append(
                    [
                        img_idx,
                        np.array([0], dtype=np.float32),
                        np.array([1], dtype=np.float32),
                        np.array([most_frequent_cat], dtype=np.int64),
                        np.array([most_frequent_supercat], dtype=np.int64),
                        np.array([dice_score], dtype=np.float32),
                    ]
                )
        return matched, all

    def forward(self, images, targets=None):
        """
        Args:
            images: Input tensor [B, C, H, W]
            targets: Ground truth bboxes per image

        Returns:
            embeddings: [B, embedding_dim]
            outputs: Per-image metrics (including losses)
            all: All predictions per image
        """

        self.model.eval()
        # Forward pass: raw predictions
        preds = self.model(images)
        
        # Compute losses if targets are provided
        losses = None
        if targets is not None:
            losses = self.compute_losses(images, targets)
        
        # Apply confidence threshold + NMS
        final_preds, keep_indices = non_max_suppression(
            preds,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=300,
            return_idxs=True,
        )
        
        # Slice raw predictions to keep logits/objectness
        batch_idx = []
        for img_idx, img in enumerate(keep_indices):
            batch_idx.extend([img_idx] * len(img))  # batch index
        batch_idx = torch.tensor(batch_idx, device=images.device)
        final_preds = torch.cat([f for f in final_preds], dim=0)
        
        # Replace targets/preds with filtered ones
        final_preds, all = self.match_targets_to_preds(
            final_preds, batch_idx, targets, len(images)
        )
        
        # Add losses to final_preds
        if losses is not None:
            for i, pred in enumerate(final_preds):
                pred.extend([
                    losses[i]['box_loss'],
                    losses[i]['cls_loss'],
                    losses[i]['dfl_loss'],
                ])
                if self.is_seg:
                    pred.append(losses[i]['seg_loss'])
        
        embeddings = torch.cat([m.features for m in self.detection_layer_entry], dim=1)

        return embeddings, final_preds, all

    def predict_and_store_raw(
        self,
        images: torch.Tensor,
        img_ids,
        emb_conn: sqlite3.Connection,
        model_name: str,
        batch_base: int = 0,
        compress: bool = True,
        insert_batch: int = 128,
    ):
        """Run the model to capture raw head outputs and store them in DB.

        This function is resilient to several ultralytics return formats:
        - If the underlying model returns per-head batched tensors (list of tensors
          each shaped [B,...]) we split them per-image.
        - If it returns a per-image iterable, we accept that.

        Args:
            images: input batch tensor [B,C,H,W]
            img_ids: iterable of image ids (len == B)
            emb_conn: sqlite3.Connection to embeddings DB
            model_name: identifier string for the model
            batch_base: offset for global indexing (not used here but kept for API)
            compress: whether to compress head blobs
            insert_batch: rows per DB transaction
        """
        self.model.eval()
        cur = emb_conn.cursor()
        B = images.shape[0]
        with torch.no_grad():
            try:
                raw_out = self.model.model(images)
            except Exception:
                raw_out = self.model(images)

        try:
            pooled = torch.cat([m.features for m in self.detection_layer_entry], dim=1)
            pooled = pooled.detach().cpu()
        except Exception:
            pooled = None

        rows = []
        for i in range(B):
            img_id = int(img_ids[i]) if hasattr(img_ids[i], "__int__") else int(img_ids[i])

            head_tensors = None
            try:
                if isinstance(raw_out, (list, tuple)) and all(
                    isinstance(t, torch.Tensor) and t.shape[0] == B for t in raw_out
                ):
                    head_tensors = [t.detach().cpu()[i] for t in raw_out]
                elif isinstance(raw_out, (list, tuple)) and len(raw_out) == B:
                    candidate = raw_out[i]
                    if isinstance(candidate, (list, tuple)):
                        head_tensors = [
                            torch.as_tensor(x).detach().cpu()
                            if not isinstance(x, torch.Tensor)
                            else x.detach().cpu()
                            for x in candidate
                        ]
                    elif isinstance(candidate, torch.Tensor):
                        head_tensors = [candidate.detach().cpu()]
                    else:
                        head_tensors = [torch.as_tensor(candidate).detach().cpu()]
                elif isinstance(raw_out, torch.Tensor) and raw_out.shape[0] == B:
                    head_tensors = [raw_out.detach().cpu()[i]]
                else:
                    layer_heads = []
                    for m in self.detection_layer_entry:
                        if hasattr(m, "pred") and isinstance(m.pred, torch.Tensor):
                            layer_heads.append(m.pred.detach().cpu())
                    if len(layer_heads) > 0 and all(h.shape[0] == B for h in layer_heads):
                        head_tensors = [h[i] for h in layer_heads]
            except Exception:
                head_tensors = None

            if head_tensors is None:
                continue

            head_blob, shapes_json, dtypes_json = tensors_to_heads_blob(head_tensors, compress=compress)
            anchors_info = json.dumps({})
            features_blob = None
            if pooled is not None:
                try:
                    feat_arr = pooled[i].numpy().astype(np.float32)
                    features_blob = array_to_blob(feat_arr)
                except Exception:
                    features_blob = None

            created_ts = int(time.time())
            rows.append(
                (
                    img_id,
                    model_name,
                    head_blob,
                    shapes_json,
                    dtypes_json,
                    anchors_info,
                    features_blob,
                    created_ts,
                )
            )

            if len(rows) >= insert_batch:
                cur.executemany(
                    "INSERT OR REPLACE INTO predictions_raw_heads (img_id, model_name, head_blob, head_shapes, head_dtypes, anchors_info, features, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                emb_conn.commit()
                rows = []

        if len(rows) > 0:
            cur.executemany(
                "INSERT OR REPLACE INTO predictions_raw_heads (img_id, model_name, head_blob, head_shapes, head_dtypes, anchors_info, features, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            emb_conn.commit()

        return {"ok": True, "message": "stored"}


def _directed_class_seg_loss(heads_pred: List[torch.Tensor], heads_tgt: List[torch.Tensor], is_seg: bool = False) -> float:
    """Compute directed loss from preds -> targets by converting target logits to pseudo-labels.

    For each corresponding head level, form binary targets = sigmoid(target_logits) > 0.5
    then compute BCEWithLogitsLoss(pred_logits, binary_targets). Sum class and seg components per level.
    """
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    total = 0.0
    n_levels = max(len(heads_pred), len(heads_tgt))
    for i in range(n_levels):
        if i >= len(heads_pred) or i >= len(heads_tgt):
            continue
        pa = heads_pred[i].detach().cpu()
        pb = heads_tgt[i].detach().cpu()

        # remove possible batch dim of 1
        if pa.dim() == 4 and pa.shape[0] == 1:
            pa = pa.squeeze(0)
        if pb.dim() == 4 and pb.shape[0] == 1:
            pb = pb.squeeze(0)

        def to_chan_first(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                return x.unsqueeze(0)
            if x.dim() == 3:
                # Heuristic: assume (C,H,W) if first dim small, else (H,W,C)
                if x.shape[0] <= 32:
                    return x
                else:
                    return x.permute(2, 0, 1)
            return x

        pa = to_chan_first(pa)
        pb = to_chan_first(pb)

        ca = pa.shape[0]
        cb = pb.shape[0]
        cmin = min(ca, cb)
        if cmin == 0:
            continue
        pa_trim = pa[:cmin]
        pb_trim = pb[:cmin]

        with torch.no_grad():
            tgt_probs = torch.sigmoid(pb_trim)
            tgt_labels = (tgt_probs > 0.5).float()

        pa_flat = pa_trim.reshape(cmin, -1).transpose(0, 1).reshape(-1)
        tgt_flat = tgt_labels.reshape(cmin, -1).transpose(0, 1).reshape(-1)

        try:
            l = float(loss_fn(pa_flat, tgt_flat))
        except Exception:
            l = float(torch.norm(pa_flat - tgt_flat).item())

        total += l

        if is_seg:
            spa_a = pa_trim.mean(dim=0)
            spa_b = pb_trim.mean(dim=0)
            spa_a_flat = spa_a.reshape(-1)
            with torch.no_grad():
                spa_tgt = (torch.sigmoid(spa_b) > 0.5).float().reshape(-1)
            try:
                lseg = float(loss_fn(spa_a_flat, spa_tgt))
            except Exception:
                lseg = float(torch.norm(spa_a_flat - spa_tgt).item())
            total += lseg

    return float(total)


def symmetric_class_seg_distance_from_heads(heads_a: List[torch.Tensor], heads_b: List[torch.Tensor], is_seg: bool = False) -> float:
    """Compute symmetric distance between two images by averaging directed losses across levels.

    distance = 0.5 * (loss(A->B) + loss(B->A)), where loss sums BCE class and seg terms per level.
    """
    la = _directed_class_seg_loss(heads_a, heads_b, is_seg=is_seg)
    lb = _directed_class_seg_loss(heads_b, heads_a, is_seg=is_seg)
    return 0.5 * (la + lb)


class Clustering_layer:
    def __init__(self, model_str, model_name, debug=False):
        self.model_str = model_str
        self.model_name = model_name
        
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
            self.model = yolowrapper(model_str)
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
        """Compute pairwise distances and store them in SQLite without loading the full matrix."""
        batch_size = len(next(iter(data_loader))[0])
        embeddings_db = self.embeddings_db
        distances_db = self.distances_db

        # --- Open embeddings DB ---
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        if pragma_speed:
            emb_conn.execute("PRAGMA journal_mode=WAL")
            emb_conn.execute("PRAGMA synchronous=OFF")
            emb_conn.execute("PRAGMA temp_store=MEMORY")
            emb_conn.execute("PRAGMA cache_size=-20000")
        emb_cursor = emb_conn.cursor()
        
        # Create table schema based on model type
        if "yolo" in self.model_name:
            if self.is_seg:
                # Segmentation model schema
                emb_cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        img_id INTEGER,
                        embedding BLOB,
                        hit_freq REAL,
                        mean_iou REAL,
                        mean_conf REAL,
                        mean_dice REAL,
                        flag_cat INTEGER,
                        flag_supercat INTEGER,
                        box_loss REAL,
                        cls_loss REAL,
                        dfl_loss REAL,
                        seg_loss REAL
                    )
                """)
                emb_cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY,
                        img_id INTEGER,
                        iou BLOB,
                        conf BLOB,
                        dice BLOB,
                        cat BLOB,
                        supercat BLOB
                    )
                """)
            else:
                # Detection model schema
                emb_cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        img_id INTEGER,
                        embedding BLOB,
                        hit_freq REAL,
                        mean_iou REAL,
                        mean_conf REAL,
                        flag_cat INTEGER,
                        flag_supercat INTEGER,
                        box_loss REAL,
                        cls_loss REAL,
                        dfl_loss REAL
                    )
                """)
                emb_cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY,
                        img_id INTEGER,
                        iou BLOB,
                        conf BLOB,
                        cat BLOB,
                        supercat BLOB
                    )
                """)
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
        emb_cursor.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value INTEGER)"
        )
        emb_conn.commit()

        # Setup distances database
        dist_conn = sqlite3.connect(distances_db)
        dist_conn.execute("PRAGMA journal_mode=WAL")
        dist_conn.execute("PRAGMA synchronous=OFF")
        dist_conn.execute("PRAGMA temp_store=MEMORY")
        dist_conn.execute("PRAGMA cache_size=-20000")
        dist_cursor = dist_conn.cursor()
        dist_cursor.execute("""
            CREATE TABLE IF NOT EXISTS distances (
                i INTEGER,
                j INTEGER,
                distance REAL,
                PRIMARY KEY (i, j)
            )
        """)
        dist_cursor.execute(
            "CREATE TABLE IF NOT EXISTS progress (last_i INTEGER PRIMARY KEY, last_j INTEGER)"
        )
        dist_cursor.execute("CREATE INDEX IF NOT EXISTS idx_i ON distances(i)")
        dist_cursor.execute("CREATE INDEX IF NOT EXISTS idx_j ON distances(j)")
        dist_conn.commit()

        # Step 1: Check and compute embeddings
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
                    emb, outputs, all_output = self.model(images, labels)
                else:
                    emb = self.model(images)

                if "yolo" in self.model_name:
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
                            "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                            "INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
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
                            "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                            "INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
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
                        "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        row,
                    )
                emb_conn.commit()
                idx += len(images)
                new_embeddings += len(images)

        if new_embeddings > 0:
            emb_cursor.execute(
                'INSERT OR REPLACE INTO metadata VALUES ("total_count", ?)', (idx,)
            )
            emb_conn.commit()
            self.logger.info(f"Added {new_embeddings} new embeddings. Total: {idx}")
        else:
            self.logger.info(f"All {existing_count} embeddings already computed")

        emb_conn.close()
        emb_conn = sqlite3.connect(embeddings_db, timeout=30)
        emb_cursor = emb_conn.cursor()

        # Ensure raw YOLO head predictions are present for all embeddings.
        # If some are missing, run the model to store them via `predict_and_store_raw`.
        if "yolo" in self.model_name:
            self.logger.info("Verifying raw head predictions for YOLO model...")
            emb_cursor.execute("SELECT id, img_id FROM embeddings")
            emb_rows = emb_cursor.fetchall()
            missing_img_ids = []
            for eid, img_id in emb_rows:
                emb_cursor.execute(
                    "SELECT 1 FROM predictions_raw_heads WHERE img_id = ? AND model_name = ?",
                    (int(img_id), self.model_name),
                )
                if emb_cursor.fetchone() is None:
                    missing_img_ids.append(int(img_id))

            if len(missing_img_ids) > 0:
                self.logger.info(f"Found {len(missing_img_ids)} missing raw heads; computing them now.")
                conn = emb_conn
                # Iterate dataset and compute raw heads for batches containing missing images
                for batch_idx, (img_ids, images, labels) in enumerate(data_loader):
                    try:
                        batch_img_ids = [int(x) for x in img_ids]
                    except Exception:
                        batch_img_ids = []
                    if not any(i in missing_img_ids for i in batch_img_ids):
                        continue
                    images = images.to(device)
                    try:
                        self.model.predict_and_store_raw(images, img_ids, emb_conn=conn, model_name=self.model_name)
                    except Exception as e:
                        self.logger.error(f"Error saving raw heads for batch {batch_idx}: {e}")
                self.logger.info("Finished writing missing raw heads.")
            else:
                self.logger.info("All raw heads already present.")

        # Step 2: Compute distances with efficient resume
        n_samples = len(data_loader.dataset)
        n_blocks = len(data_loader)

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
                    start = int(block_idx * batch_size)
                    end = int(min((block_idx + 1) * batch_size, n_samples))
                    emb_cursor.execute(
                        "SELECT id, img_id FROM embeddings WHERE id >= ? AND id < ?",
                        (start, end),
                    )
                    rows_local = emb_cursor.fetchall()
                    id_to_heads = {}
                    ids = []
                    for eid, img_id in rows_local:
                        emb_cursor.execute(
                            "SELECT head_blob FROM predictions_raw_heads WHERE img_id = ? AND model_name = ?",
                            (int(img_id), self.model_name),
                        )
                        r = emb_cursor.fetchone()
                        if r is None:
                            raise RuntimeError(f"Missing raw heads for img_id={img_id} (embedding id={eid})")
                        head_blob = r[0]
                        # strictly require correct deserialization
                        heads = heads_blob_to_tensors(head_blob)
                        id_to_heads[int(eid)] = heads
                        ids.append(int(eid))
                    return ids, id_to_heads

                # preload heads for i_block
                idx_i_heads, id2heads_i = load_head_block_map(i_block)

                for j_block in range(i_block, n_blocks):
                    idx_j_heads, id2heads_j = load_head_block_map(j_block)
                    if len(idx_j_heads) == 0:
                        raise RuntimeError(f"Empty head block at j_block={j_block}")

                    rows = []
                    for gid_i in idx_i_heads:
                        if gid_i not in id2heads_i:
                            raise RuntimeError(f"Missing heads for embedding id {gid_i} in block {i_block}")
                        heads_i = id2heads_i[gid_i]
                        for gid_j in idx_j_heads:
                            # enforce upper triangle ordering
                            if i_block == j_block and gid_j < gid_i:
                                continue
                            if gid_j not in id2heads_j:
                                raise RuntimeError(f"Missing heads for embedding id {gid_j} in block {j_block}")
                            heads_j = id2heads_j[gid_j]
                            # compute symmetric distance (may raise)
                            d = symmetric_class_seg_distance_from_heads(heads_i, heads_j, is_seg=self.is_seg)
                            rows.append((int(gid_i), int(gid_j), float(d)))

                    if rows:
                        dist_cursor.executemany(
                            "INSERT OR REPLACE INTO distances (i, j, distance) VALUES (?, ?, ?)",
                            rows,
                        )
                    dist_conn.commit()
                self.logger.debug(f"Completed head-based block row {i_block + 1}/{n_blocks}")
                # continue to next i_block
                continue

            # Fallback: compute distances from embeddings (existing behavior)
            for j_block in range(i_block, n_blocks):  # only upper triangle
                idx_j, emb_j = load_block(j_block)
                if emb_j is None:
                    self.logger.info(f"Skipping empty column {j_block}")
                    continue

                dist_block = torch.cdist(emb_i, emb_j, p=2).cpu().numpy()

                rows = np.column_stack([
                    np.repeat([gid for gid in idx_i], len(idx_j)),
                    np.tile([gid for gid in idx_j], len(idx_i)),
                    dist_block.ravel(),
                ]).tolist()

                dist_cursor.executemany(
                    "INSERT OR REPLACE INTO distances (i, j, distance) VALUES (?, ?, ?)",
                    rows,
                )

            dist_conn.commit()
            self.logger.debug(f"Completed block row {i_block + 1}/{n_blocks}")

        self.logger.info("Distance computation complete")
        emb_conn.close()
        dist_conn.close()
        return embeddings_db, distances_db