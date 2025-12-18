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


class yolowrapper(torch.nn.Module):
    def __init__(self, model_str):
        super().__init__()
        self.model = YOLO(model_str).model
        self.model.eval()
        self.model.to(device)
        self.model.device = device

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

    def match_targets_to_preds(
        self, final_preds, batch_idx, targets, N, iou_thres=0.6, conf_thres=0.5
    ):
        """
        final_preds: list of tensors per image, each [N,6] (x1,y1,x2,y2,conf,cls)
        batch_idx: indexes of predicted boxes in the batch
        targets: dict with GT boxes and labels
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
            if mask.sum() == 0:
                if len(preds) == 0:
                    miou = 1
                    cat = -1
                    hitfreq = 1
                else:
                    miou = 0
                    cat = -2
                    hitfreq = 0
                matched.append([miou, 1, cat, cat, hitfreq])
                all.append(
                    [
                        img_idx,
                        np.array([miou], dtype=np.float32),
                        np.array([1], dtype=np.float32),
                        np.array([cat], dtype=np.float32),
                        np.array([cat], dtype=np.float32),
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
                    ]
                )
                all.append(all_match)
            else:
                matched.append([0, 1, most_frequent_cat, most_frequent_supercat, 0])
                all.append(
                    [
                        img_idx,
                        np.array([0], dtype=np.float32),
                        np.array([1], dtype=np.float32),
                        np.array([most_frequent_cat], dtype=np.float32),
                        np.array([most_frequent_supercat], dtype=np.float32),
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
            losses (optional): List of per-image losses, where each is:
                                [[box_loss, cls_loss, dfl_loss], ...] per bbox
        """

        self.model.eval()
        # Forward pass: raw predictions + loss items
        preds = self.model(images)
        # Apply confidence threshold + NMS
        final_preds, keep_indices = non_max_suppression(
            preds,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=300,
            return_idxs=True,  # you need to modify NMS to give you this
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
        embeddings = torch.cat([m.features for m in self.detection_layer_entry], dim=1)

        return embeddings, final_preds, all


class Clustering_layer:
    def __init__(self, model_str, model_name, debug=False):
        self.model_str = model_str
        self.model_name = model_name

        if model_str == "dino":
            self.model = DINO().to(device)
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
        """Compute pairwise distances and store them in SQLite without loading the full matrix. - Embeddings are inserted in batches (executemany). - Distances are computed block-by-block and written in bulk. - Progress table allows safe resume after interruption."""
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

        idx = existing_count
        new_embeddings = 0
        if existing_count < len(data_loader.dataset):
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
                    for id, (img_id, vec, output) in enumerate(
                        zip(img_ids, emb.cpu(), outputs)
                    ):
                        miou, mconf, cat, supercat, hit_freq = output
                        row.append(
                            [
                                int((batch_ids * batch_size) + id),
                                int(img_id),
                                vec.numpy().tobytes(),
                                hit_freq,
                                miou,
                                mconf,
                                cat,
                                supercat,
                            ]
                        )
                    emb_cursor.executemany(
                        "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        row,
                    )
                    for id, (img_id, miou, mconf, cat, supercat) in enumerate(
                        all_output
                    ):
                        all_row.append(
                            [
                                int(batch_ids * batch_size + id),
                                int(img_id),
                                miou.tobytes(),
                                mconf.tobytes(),
                                cat.tobytes(),
                                supercat.tobytes(),
                            ]
                        )
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
                        row.append(
                            [
                                int((batch_ids * batch_size) + id),
                                int(img_id),
                                vec.numpy().tobytes(),
                                0,
                                0,
                                0,
                                most_frequent_cat,
                                most_frequent_supercat,
                            ]
                        )
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

            for j_block in range(i_block, n_blocks):  # only upper triangle
                idx_j, emb_j = load_block(j_block)
                if emb_j is None:
                    self.logger.info(f"Skipping empty column {j_block}")
                    continue

                dist_block = torch.cdist(emb_i, emb_j, p=2).cpu().numpy()

                rows = np.column_stack(
                    [
                        np.repeat([gid for gid in idx_i], len(idx_j)),
                        np.tile([gid for gid in idx_j], len(idx_i)),
                        dist_block.ravel(),
                    ]
                ).tolist()

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
