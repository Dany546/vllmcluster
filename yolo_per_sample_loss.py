"""Wrapper to use custom ultralytics with per-sample loss computation.

This module allows YOLOExtractor to get per-sample losses instead of batch-aggregated ones,
enabling fast pairwise distance computation.
"""

import torch
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add custom ultralytics to path so it overrides the installed version
CUSTOM_ULTRALYTICS_PATH = Path(__file__).parent / "ultralytics_custom"
if str(CUSTOM_ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(CUSTOM_ULTRALYTICS_PATH))


class PerSampleLossWrapper:
    """Wrapper around YOLO model to extract per-sample losses."""
    
    def __init__(self, yolo_model):
        """Initialize with a YOLO model instance."""
        self.model = yolo_model
        self.device = next(self.model.model.parameters()).device
        
        # Enable per-sample loss mode in the criterion
        if hasattr(self.model.model, 'criterion'):
            self.model.model.criterion.return_per_sample_loss = True
    
    def compute_losses_for_pairs(
        self,
        preds_i: List[Dict[str, torch.Tensor]],
        images_j: torch.Tensor,
        targets_j_raw: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-pair loss matrix efficiently.
        
        Args:
            preds_i: List of prediction dicts from images in block i
            images_j: Tensor of images from block j, shape [n_j, C, H, W]
            targets_j_raw: Raw targets dict with 'bboxes', 'cls', 'batch_idx'
        
        Returns:
            loss_matrix: Tensor of shape [n_i, n_j, 3] for detection (or [n_i, n_j, 4] for seg)
                Each [i, j, :] = [box_loss, cls_loss, dfl_loss] (or with seg_loss)
        """
        n_i = len(preds_i)
        n_j = images_j.shape[0]
        n_comps = 4 if 'seg' in self.model.model.__class__.__name__ else 3
        
        loss_matrix = torch.zeros(n_i, n_j, n_comps, device=self.device, dtype=torch.float32)
        
        was_training = self.model.model.training
        self.model.model.train()
        
        try:
            for i in range(n_i):
                pred_i = preds_i[i]
                boxes_i = pred_i.get('boxes', torch.tensor([], device=self.device))
                classes_i = pred_i.get('classes', torch.tensor([], device=self.device))
                
                if len(boxes_i) == 0:
                    continue
                
                # For each image j, compute loss of predictions from i on image j
                for j in range(n_j):
                    img_j = images_j[j : j + 1]  # [1, C, H, W]
                    
                    # Build targets for this pair
                    batch_targets = {
                        'bboxes': boxes_i.to(self.device).float(),
                        'cls': classes_i.to(self.device).long(),
                        'batch_idx': torch.zeros(len(boxes_i), device=self.device, dtype=torch.long),
                        'img': img_j,
                    }
                    
                    try:
                        # Forward pass in training mode
                        out = self.model.model(batch_targets)
                        if isinstance(out, tuple):
                            loss, loss_items = out
                        else:
                            loss = out
                            loss_items = None
                        
                        # Extract per-sample losses
                        if loss_items is not None and hasattr(loss_items, 'shape'):
                            if n_comps == 3:
                                # Detection: [box, cls, dfl]
                                if loss_items.shape[0] >= 3:
                                    loss_matrix[i, j, 0] = loss_items[0].item()
                                    loss_matrix[i, j, 1] = loss_items[1].item()
                                    loss_matrix[i, j, 2] = loss_items[2].item()
                            else:
                                # Segmentation: [box, seg, cls, dfl]
                                if loss_items.shape[0] >= 4:
                                    loss_matrix[i, j, 0] = loss_items[0].item()
                                    loss_matrix[i, j, 1] = loss_items[1].item()
                                    loss_matrix[i, j, 2] = loss_items[2].item()
                                    loss_matrix[i, j, 3] = loss_items[3].item()
                    except Exception as e:
                        pass  # Leave as zero if computation fails
        finally:
            if not was_training:
                self.model.model.eval()
        
        return loss_matrix
