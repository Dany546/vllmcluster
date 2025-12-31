import logging
import sqlite3
from pydoc import classify_class_attrs

import numpy as np
import torch
import torch.nn.functional as F
from cfg import cat_to_super, device
from model import CLIP, DINO
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
            f"/globalscratch/ucl/irec/darimez/dino/embeddings/{model_name}.db"
        )
        self.distances_db = (
            f"/globalscratch/ucl/irec/darimez/dino/distances/{model_name}.db"
        )

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def distance_matrix_db(
        self,
        data_loader,
        pragma_speed: bool = True,
    ) -> Tuple[str, str]:
        """Compute pairwise distances and store embeddings in a vector-enabled SQLite DB."""
        batch_size = len(next(iter(data_loader))[0])
        embeddings_db = self.embeddings_db
        distances_db = self.distances_db  # you can keep this path if you want to store other info

        # --- Open vector-enabled embeddings DB ---
        from sqlvector_utils import connect_vec_db, create_embeddings_table, serialize_float32_array, insert_embeddings_batch, build_vector_index

        emb_conn = connect_vec_db(embeddings_db)
        # dimension: infer from model output by running one batch or set explicitly
        # Here we infer from a dummy forward pass if possible
        # We'll assume model returns embeddings of shape [B, D] when called with images
        # Use a small sample to infer dim
        img_ids, sample_images, labels = next(iter(data_loader))
        sample_images = sample_images.to(device) 
        if "yolo" in self.model_name:
            sample_emb, _, _ = self.model(sample_images, labels)
            dim = sample_emb.shape[1]
        else:
            sample_emb = self.model(sample_images)
            dim = sample_emb.shape[1]

        create_embeddings_table(emb_conn, dim, is_seg=self.is_seg)

        # --- Insert embeddings while regenerating them ---
        emb_conn.close()
        emb_conn = connect_vec_db(embeddings_db)
        total_inserted = 0
        for batch_ids, (img_ids, images, labels) in enumerate(data_loader):
            images = images.to(device)
            if "yolo" in self.model_name:
                emb, outputs, all_output = self.model(images, labels)
            else:
                emb = self.model(images)

            rows = []
            if "yolo" in self.model_name:
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

        # Build index once after all inserts
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
        dist_conn.close()

        self.logger.info(f"Inserted {total_inserted} embeddings into {embeddings_db} and built index")
        return embeddings_db, distances_db
