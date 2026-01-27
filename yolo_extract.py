import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from pathlib import Path
import cv2
import numpy as np
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

def xywh_norm_to_xyxy_abs(boxes, img_width, img_height):
    """
    Convert normalized xywh (center format) to absolute xyxy format.
    
    Args:
        boxes (torch.Tensor): Bounding boxes in normalized xywh format [x_center, y_center, width, height]
                              where all values are in [0, 1]
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
    
    Returns:
        torch.Tensor: Bounding boxes in absolute xyxy format [x1, y1, x2, y2]
    """
    if boxes.numel() == 0:
        return boxes
    
    boxes = boxes.clone()
    # Convert from normalized to absolute
    boxes[:, 0] = boxes[:, 0] * img_width   # x_center
    boxes[:, 1] = boxes[:, 1] * img_height  # y_center
    boxes[:, 2] = boxes[:, 2] * img_width   # width
    boxes[:, 3] = boxes[:, 3] * img_height  # height
    
    # Convert from xywh (center) to xyxy (corners)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)


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


class YOLOExtractor:
    def __init__(self, model_name='yolo11x.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize YOLO model for extraction.
        
        Args:
            model_name: 'yolo11x.pt' or 'yolo11x-seg.pt'
            device: 'cuda' or 'cpu'
        """
        self.model = YOLO(model_name)
        self.device = device
        self.is_seg = 'seg' in model_name
        
        # Initialize model args properly to avoid AttributeError in loss computation
        args = self.model.model.args
        args['overlap_mask'] = True
        args['mosaic'] = 0
        args['profile'] = False
        for key in ['box', 'cls', 'dfl']:
            args[key] = 1
        self.model.model.args = SimpleNamespace(**args)
        
        # Move model to device
        self.model.to(device)
        self.model.model.device = device
        
        # Initialize criterion for loss computation
        self.model.model.criterion = self.model.model.init_criterion()
        # Enable per-sample loss mode
        self.model.model.criterion.return_per_sample_loss = True
        self.model.model.eval()
        
        self.detection_layer_entry = None
        self._register_hooks()
        self.debug = False  # Can be set externally for logging
        
    def _register_hooks(self):
        """Register forward hooks to capture GAP features from inputs to detection head cv2 layers."""
        def hook_fn(module, input, output):
            # Capture input features and apply GAP
            input_tensor = input[0] if isinstance(input, tuple) else input
            module.features = (
                torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )
        
        # Register hooks on inputs of detection heads (cv2 layers)
        if hasattr(self.model.model, 'model'):
            self.detection_layer_entry = self.model.model.model[-1].cv2
            for m in self.detection_layer_entry:
                m.features = None
                m.register_forward_hook(hook_fn)
    
    def extract_head_features(self, image_path, use_gap=True):
        """
        Deprecated: Use run_with_predictor instead.
        This is kept for backward compatibility.
        """
        self.model.model.eval()
        _ = self.model(image_path, save=False, verbose=False)
        
        features = {}
        if self.detection_layer_entry is not None:
            for i, m in enumerate(self.detection_layer_entry):
                if hasattr(m, 'features') and m.features is not None:
                    features[f'cv2_{i}'] = m.features.cpu().detach()
        return features
        
    def extract_predictions(self, image_path):
        """
        Extract predictions in eval mode.
        
        Returns:
            dict: Contains boxes, classes, confidences, and masks (if seg model)
        """
        self.model.model.eval()
        
        results = self.model(image_path, save=False, verbose=False)[0]
        
        predictions = {
            'boxes': results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([]),
            'classes': results.boxes.cls.cpu().numpy() if results.boxes is not None else np.array([]),
            'confidences': results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([]),
            'class_names': [results.names[int(c)] for c in results.boxes.cls] if results.boxes is not None else [],
        }
        
        if self.is_seg and results.masks is not None:
            predictions['masks'] = results.masks.data.cpu().numpy()
            predictions['mask_shape'] = results.masks.orig_shape
        
        return predictions
    
    def extract_losses(self, image_path, label_path=None):
        """
        Extract loss components in train mode.
        
        Args:
            image_path: Path to image
            label_path: Path to label file (YOLO format). If None, assumes same dir/name as image
            
        Returns:
            dict: Contains all loss components
        """
        self.model.model.train()
        
        # Prepare image
        img = cv2.imread(str(image_path))
        img_tensor = self.model.predictor.preprocess(img)
        img_tensor = img_tensor.to(self.device)
        
        # Prepare labels if provided
        if label_path is None:
            # Try to find label file with same name
            img_path = Path(image_path)
            label_path = img_path.parent.parent / 'labels' / img_path.stem
            label_path = f"{label_path}.txt"
        
        labels = self._load_labels(label_path, img.shape)
        
        # Forward pass
        with torch.set_grad_enabled(True):
            preds = self.model.model(img_tensor)
            
            # Calculate losses
            loss_dict = {}
            if hasattr(self.model.model, 'criterion'):
                # Get loss from criterion
                losses = self.model.model.criterion(preds, labels)
                
                if isinstance(losses, dict):
                    loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                               for k, v in losses.items()}
                else:
                    # If single loss value, try to extract components
                    loss_dict['total_loss'] = losses.item() if isinstance(losses, torch.Tensor) else losses
            else:
                # Manual loss extraction from predictions
                loss_dict['note'] = 'Loss criterion not directly accessible'
        
        # Also get raw predictions
        raw_preds = {
            'pred_boxes': preds[0].cpu().detach().numpy() if isinstance(preds, (list, tuple)) else None,
            'pred_shape': preds[0].shape if isinstance(preds, (list, tuple)) else None,
        }
        
        loss_dict['raw_predictions'] = raw_preds
        
        return loss_dict

    # --- Predictor-driven pipeline ---
    def _to_tensor_batch(self, source: Union[Sequence[Any], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Normalize diverse YOLO sources into a BCHW float tensor on the configured device."""
        if isinstance(source, torch.Tensor):
            t = source
            if t.dim() == 3:
                t = t.unsqueeze(0)
            return t.to(self.device, dtype=torch.float32)
        if isinstance(source, np.ndarray) and source.ndim == 3:
            return torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        if isinstance(source, (list, tuple)):
            frames = []
            for img in source:
                if isinstance(img, torch.Tensor):
                    f = img
                else:
                    f = torch.from_numpy(np.asarray(img))
                if f.dim() == 3 and f.shape[0] in (1, 3):
                    pass  # already CHW
                elif f.dim() == 3:
                    f = f.permute(2, 0, 1)
                else:
                    raise ValueError("Unsupported image shape for conversion to tensor")
                frames.append(f.float())
            return torch.stack(frames, dim=0).to(self.device)
        raise ValueError("Unsupported source type for tensor conversion")

    def _compute_losses(self, images: torch.Tensor, targets: Dict[str, Any]) -> List[Dict[str, float]]:
        """Compute per-image loss components using the model's criterion in training mode."""
        was_training = self.model.model.training
        self.model.model.train()

        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
        batch['img'] = images

        loss_dict: List[Dict[str, float]] = []
        try:
            out = self.model.model(batch)
            if isinstance(out, tuple):
                loss, loss_items = out
            else:
                loss = out
                loss_items = None

            bs = images.shape[0]
            if self.is_seg:
                # seg models return box, seg, cls, dfl
                if loss_items is not None and len(loss_items) >= 4:
                    box_loss, seg_loss, cls_loss, dfl_loss = [x.item() / bs for x in loss_items[:4]]
                else:
                    box_loss = seg_loss = cls_loss = dfl_loss = float(loss.item()) / bs if hasattr(loss, 'item') else 0.0
                for _ in range(bs):
                    loss_dict.append({
                        'box_loss': box_loss,
                        'seg_loss': seg_loss,
                        'cls_loss': cls_loss,
                        'dfl_loss': dfl_loss,
                    })
            else:
                if loss_items is not None and len(loss_items) >= 3:
                    box_loss, cls_loss, dfl_loss = [x.item() / bs for x in loss_items[:3]]
                else:
                    box_loss = cls_loss = dfl_loss = float(loss.item()) / bs if hasattr(loss, 'item') else 0.0
                for _ in range(bs):
                    loss_dict.append({
                        'box_loss': box_loss,
                        'cls_loss': cls_loss,
                        'dfl_loss': dfl_loss,
                    })
        finally:
            if not was_training:
                self.model.model.eval()

        return loss_dict

    def run_with_predictor(
        self,
        source: Union[str, Path, Sequence[Any], torch.Tensor, np.ndarray],
        targets: Optional[Dict[str, Any]] = None,
        conf: float = 0.5,
        iou: float = 0.45,
        half: bool = False,
        embed_layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run Ultralytics Detection/SegmentationPredictor to get predictions, features, and optional losses.

        Args:
            source: Image path, tensor, numpy array, or list of those.
            targets: Optional training targets dict (expects ultralytics keys: bboxes, cls, batch_idx, masks).
            conf: Confidence threshold for predictor.
            iou: IoU threshold for predictor NMS.
            half: Use half precision if supported.
            embed_layers: Layer indices for predictor embed output (default grabs bottleneck with -2).

        Returns:
            dict with keys: predictions (structured tensors), results (raw Results), losses, gap_features, bottleneck_features.
        """

        predictor_cls = SegmentationPredictor if self.is_seg else DetectionPredictor
        overrides = {
            'model': self.model.model,
            'device': self.device,
            'conf': conf,
            'iou': iou,
            'half': half,
            'save': False,
            'verbose': False, 
            'imgsz': 640,  # Tell predictor images are already 640x640
        }
        if self.is_seg:
            overrides['retina_masks'] = True
        overrides['embed'] = embed_layers if embed_layers is not None else [-2]

        expected_batch_size = source.shape[0]
        self.model.model.eval()
        predictor = predictor_cls(overrides=overrides)
        predictor.model = self.model.model
        predictor.batch = [["placeholder_path.jpg" for _ in range(expected_batch_size)], None, None]  # Dummy to avoid errors  
        
        # According to Ultralytics docs, predictor natively handles:
        # - torch.Tensor: shape [B, C, H, W], BCHW format, RGB channels, float32 (0.0-1.0)
        # - Returns list of B Results objects

        # Call predictor - should return list of Results objects (one per image)
        pred_output = predictor.inference(source)
        pred_output = predictor.postprocess(pred_output, source, source)
        
        # Check if this is a list and what's inside
        if isinstance(pred_output, list):
            if len(pred_output) > 0:
                # Check if this is a list of Results objects or a list of tensors
                if hasattr(pred_output[0], 'boxes'):
                    # This is the expected format: list of Results objects
                    results = pred_output
                else:
                    raise ValueError("Unexpected predictor output format: list does not contain Results objects")
            else:
                results = []
        else:
            results = list(pred_output) if not isinstance(pred_output, list) else pred_output
        
        # Final debug: show what we have
        if self.debug:
            print(f"[DEBUG] Final results: {len(results)} objects")
        if results and hasattr(results[0], 'boxes'):
            det_counts = [r.boxes.shape[0] if r.boxes is not None else 0 for r in results[:min(5, len(results))]]
        
        # NOTE: Gap features from hooks will only contain the last batch's features
        # since the predictor processes the entire batch in one forward pass.
        # This is a known limitation when using batch tensors with hooks.

        # Pad results if we got fewer than expected (this prevents "missing prediction" errors)
        if expected_batch_size is not None and len(results) < expected_batch_size:
            missing_count = expected_batch_size - len(results)
            # Create placeholder empty results to maintain batch alignment
            for _ in range(missing_count):
                results.append(None)  # Will be converted to empty structured_pred below

        structured_preds: List[Dict[str, Any]] = []
        for r in results:
            if r is None:
                # Handle padding: create an empty prediction for missing results
                structured_preds.append({
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros((0,)),
                    'classes': torch.zeros((0,)),
                    'masks': None,
                    'embeds': [],
                })
            else:
                boxes = r.boxes.xyxy if getattr(r, 'boxes', None) is not None else torch.zeros((0, 4))
                scores = r.boxes.conf if getattr(r, 'boxes', None) is not None else torch.zeros((0,))
                classes = r.boxes.cls if getattr(r, 'boxes', None) is not None else torch.zeros((0,))
                masks = r.masks.data if self.is_seg and getattr(r, 'masks', None) is not None else None
                embeds = [e for e in getattr(r, 'embed', [])] if hasattr(r, 'embed') and r.embed is not None else []
                structured_preds.append({
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'masks': masks,
                    'embeds': embeds,
                })

        # Collect GAP features from cv2 layers
        # Note: When processing batches, hooks capture features for the entire batch
        gap_features = {}
        if self.detection_layer_entry is not None:
            for i, m in enumerate(self.detection_layer_entry):
                if hasattr(m, 'features') and m.features is not None:
                    gap_features[f'cv2_{i}'] = m.features.detach().cpu()
        
        bottleneck_features = gap_features  # For backward compatibility

        losses = None
        if targets is not None:
            img_tensor = self._to_tensor_batch(source)
            losses = self._compute_losses(img_tensor, targets)

        return {
            'predictions': structured_preds,
            'results': results,
            'losses': losses,
            'gap_features': gap_features,
            'bottleneck_features': bottleneck_features,
        }

    def labels_to_targets(self, labels: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        """Convert dataloader labels dict to ultralytics targets format for loss computation.
        
        Expected input keys: 'bboxes', 'cls', 'batch_idx', optionally 'masks'
        Returns dict with keys: 'bboxes', 'cls', 'batch_idx', 'masks' (if seg model)
        """
        targets = {}
        for key in ['bboxes', 'cls', 'batch_idx']:
            if key in labels:
                val = labels[key]
                if isinstance(val, torch.Tensor):
                    targets[key] = val.to(self.device)
                else:
                    targets[key] = torch.as_tensor(val, device=self.device)
            else:
                # Create empty tensors if missing
                if key == 'bboxes':
                    targets[key] = torch.zeros((0, 4), device=self.device, dtype=torch.float32)
                elif key == 'cls':
                    targets[key] = torch.zeros((0,), device=self.device, dtype=torch.int64)
                elif key == 'batch_idx':
                    targets[key] = torch.zeros((0,), device=self.device, dtype=torch.int64)
        
        # Handle masks for segmentation
        if self.is_seg and 'masks' in labels:
            val = labels['masks']
            if isinstance(val, torch.Tensor):
                targets['masks'] = val.to(self.device)
            else:
                targets['masks'] = torch.as_tensor(val, device=self.device)
        else:
            targets['masks'] = None
        
        return targets

    def process_yolo_batch(
        self,
        structured_preds: List[Dict[str, Any]],
        targets: Optional[Dict[str, Any]],
        original_labels: Optional[Dict[str, Any]],
        losses: Optional[List[Dict[str, float]]],
        embeddings: torch.Tensor,
        cat_to_super: Optional[Dict[int, int]] = None,
    ) -> Tuple[List, List]:
        """Process YOLO predictions and losses into format compatible with DB storage.
        
        Args:
            structured_preds: List of prediction dicts from run_with_predictor
            targets: Targets dict in ultralytics format
            original_labels: Original label info (unused, for API compat)
            losses: List of per-image loss dicts
            embeddings: Embedding tensor [B, D]
            cat_to_super: Mapping from category to supercategory (optional)
        
        Returns:
            (outputs, all_output) tuple where:
            - outputs: List of per-image metrics [miou, mconf, cat, supercat, hit_freq, dice, box_loss, cls_loss, dfl_loss, seg_loss]
            - all_output: List of per-image detailed predictions for all detections
        """
        if cat_to_super is None:
            cat_to_super = {}
        
        outputs = []
        all_output = []
        
        for img_idx, pred in enumerate(structured_preds):
            boxes = pred['boxes']  # [N, 4] xyxy
            scores = pred['scores']  # [N]
            classes = pred['classes']  # [N]
            masks = pred.get('masks')  # [N, H, W] if seg
            
            # Get loss components if available
            loss_dict = losses[img_idx] if losses and img_idx < len(losses) else {}
            box_loss = float(loss_dict.get('box_loss', 0.0))
            cls_loss = float(loss_dict.get('cls_loss', 0.0))
            dfl_loss = float(loss_dict.get('dfl_loss', 0.0))
            seg_loss = float(loss_dict.get('seg_loss', 0.0))
            
            # Get GT labels for this image
            if targets is not None:
                mask = targets['batch_idx'] == img_idx
                gt_bboxes_norm = targets['bboxes'][mask].to(self.device)  # Normalized xywh format
                gt_classes = targets['cls'][mask].to(self.device)
                
                # Convert GT bboxes from normalized xywh to absolute xyxy format
                # Predictions from YOLO are in xyxy pixel coordinates (for 640x640 images)
                # GT labels are in normalized xywh format (center, width, height in 0-1)
                # We need to convert GT to match predictions for IoU computation
                img_size = 640  # Images are preprocessed to 640x640
                gt_bboxes = xywh_norm_to_xyxy_abs(gt_bboxes_norm, img_size, img_size)
            else:
                gt_bboxes = torch.zeros((0, 4), device=self.device)
                gt_classes = torch.zeros((0,), device=self.device, dtype=torch.int64)
            
            # Validate predictions
            if boxes.numel() == 0 and gt_bboxes.numel() == 0:
                # True negative: no predictions, no ground truth
                miou = 1.0
                mconf = 1.0
                hit_freq = 1.0
                most_freq_cat = -1
                most_freq_supercat = -1
                dice_score = 1.0
                
                if self.debug:
                    print(f"[DEBUG] Image {img_idx}: TN (no pred, no GT)")
            elif boxes.numel() == 0:
                # False negative: ground truth exists but no predictions
                miou = 0.0
                mconf = 0.0
                hit_freq = 0.0
                cpu_classes = gt_classes.cpu().numpy()
                values, counts = np.unique(cpu_classes, return_counts=True)
                most_freq_cat = int(values[np.argmax(counts)])
                most_freq_supercat = int(cat_to_super.get(most_freq_cat, -1))
                dice_score = 0.0
                
                if self.debug:
                    print(f"[DEBUG] Image {img_idx}: FN (no pred, {len(gt_classes)} GT)")
                    # Additional diagnostic: show GT classes and that there were no preds
                    print(f"[DEBUG]   GT classes: {cpu_classes}, GT bboxes: {gt_bboxes}")
            else:
                # Compute matching between predictions and GT
                matches, iou_scores = [], []
                for gt_idx, gt_box in enumerate(gt_bboxes):
                    if len(boxes) == 0:
                        break
                    # Compute IoU with all predictions
                    ious = box_iou(boxes, gt_box.unsqueeze(0))  # [N_pred]
                    best_iou, best_pred_idx = ious.max(0)
                    if best_iou > 0.5:  # IoU threshold
                        matches.append({
                            'gt_idx': gt_idx,
                            'pred_idx': best_pred_idx.item(),
                            'iou': best_iou.item(),
                            'conf': scores[best_pred_idx].item(),
                        })
                        iou_scores.append(best_iou.item())
                
                if len(matches) > 0:
                    miou = np.mean(iou_scores)
                    mconf = np.mean([m['conf'] for m in matches])
                    hit_freq = len(matches) / len(gt_bboxes)
                else:
                    miou = 0.0
                    mconf = scores.mean().item() if len(scores) > 0 else 0.0
                    hit_freq = 0.0
                    # Diagnostic when GT present but no matches found
                    if self.debug and gt_bboxes.numel() > 0:
                        print(f"[DEBUG] Image {img_idx}: GT present ({len(gt_bboxes)}), but no matches -> hit_freq=0")
                        print(f"[DEBUG]   preds: {boxes} boxes")
                        print(f"[DEBUG]   GT bboxes: {gt_bboxes}")
                
                # No matches found; values above already set (miou=0.0, mconf=scores.mean(), hit_freq=0.0)
                
                # Compute Dice for segmentation
                dice_score = 0.0
                if self.is_seg and masks is not None and len(matches) > 0:
                    if 'masks' in targets and targets['masks'] is not None:
                        dice_scores = []
                        for m in matches:
                            pred_mask = masks[m['pred_idx']]
                            gt_mask = targets['masks'][targets['batch_idx'] == img_idx][m['gt_idx']]
                            dice = compute_dice_score(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                            dice_scores.append(dice.item())
                        dice_score = np.mean(dice_scores)
                
                # Get most frequent category from GT
                cpu_classes = gt_classes.cpu().numpy()
                if len(cpu_classes) > 0:
                    values, counts = np.unique(cpu_classes, return_counts=True)
                    most_freq_cat = int(values[np.argmax(counts)])
                    most_freq_supercat = int(cat_to_super.get(most_freq_cat, -1))
                else:
                    most_freq_cat = -1
                    most_freq_supercat = -1
                
                if self.debug:
                    print(f"[DEBUG] Image {img_idx}: {len(boxes)} pred, {len(gt_bboxes)} GT, "
                          f"miou={miou:.3f}, mconf={mconf:.3f}, hit_freq={hit_freq:.3f}, dice={dice_score:.3f}")
            
            # Append output for DB storage
            output = [miou, mconf, most_freq_cat, most_freq_supercat, hit_freq, dice_score,
                     box_loss, cls_loss, dfl_loss]
            if self.is_seg:
                output.append(seg_loss)
            outputs.append(output)
            
            # Append all detections for this image
            all_output.append([img_idx, 
                             np.array([miou], dtype=np.float32),
                             np.array([mconf], dtype=np.float32),
                             np.array([most_freq_cat], dtype=np.int64),
                             np.array([most_freq_supercat], dtype=np.int64),
                             np.array([dice_score], dtype=np.float32)])
        
        return outputs, all_output


    def _load_labels(self, label_path, img_shape):
        """Load YOLO format labels and convert to model format."""
        labels_data = []
        
        if Path(label_path).exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # For segmentation models, there might be additional polygon points
                        if self.is_seg and len(parts) > 5:
                            poly_points = list(map(float, parts[5:]))
                            labels_data.append([cls, x_center, y_center, width, height] + poly_points)
                        else:
                            labels_data.append([cls, x_center, y_center, width, height])
        
        # Convert to tensor format expected by model
        if labels_data:
            labels_tensor = torch.tensor(labels_data, device=self.device)
            # Format: batch_idx, class, x_center, y_center, width, height
            batch_idx = torch.zeros((labels_tensor.shape[0], 1), device=self.device)
            labels = torch.cat([batch_idx, labels_tensor], dim=1)
        else:
            labels = torch.zeros((0, 6), device=self.device)
        
        return labels
    
    def predictions_to_yolo_labels(self, predictions, img_shape, conf_threshold=0.25):
        """
        Convert predictions to YOLO label format (normalized coordinates).
        
        Args:
            predictions: Output from extract_predictions()
            img_shape: (height, width) of the image
            conf_threshold: Minimum confidence to include detection
            
        Returns:
            list: YOLO format labels [class, x_center, y_center, width, height]
                  For seg models: [class, x_center, y_center, width, height, *polygon_points]
        """
        h, w = img_shape[:2]
        yolo_labels = []
        
        boxes = predictions.get('boxes')
        classes = predictions.get('classes')
        confs = predictions.get('confidences', predictions.get('scores'))
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
            if conf < conf_threshold:
                continue
            
            # Convert xyxy to xywh normalized
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            label = [int(cls), x_center, y_center, width, height]
            
            # Add segmentation polygon if available
            if self.is_seg and 'masks' in predictions and predictions['masks'] is not None:
                mask = predictions['masks'][i]
                # Convert mask to polygon points (normalized)
                contours = self._mask_to_polygon(mask, (h, w))
                if contours:
                    label.extend(contours)
            
            yolo_labels.append(label)
        
        return yolo_labels
    
    def _mask_to_polygon(self, mask, img_shape):
        """Convert binary mask to normalized polygon coordinates."""
        h, w = img_shape
        
        # Resize mask to image size if needed
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Flatten and normalize
        points = contour.reshape(-1, 2)
        normalized = []
        for x, y in points:
            normalized.extend([x / w, y / h])
        
        return normalized
    
    def save_yolo_labels(self, predictions, img_path, img_shape, output_dir=None, conf_threshold=0.25, do_save=False):
        """
        Save predictions as YOLO label file. Writing is opt-in: set `do_save=True` and provide `output_dir` to write a label file.
        
        Args:
            predictions: Output from extract_predictions()
            img_path: Path to source image
            img_shape: (height, width) of the image
            output_dir: Directory to save label file (required if do_save=True)
            conf_threshold: Minimum confidence to include
            do_save: If True, actually write the label file. Default: False (no disk writes).
        """
        yolo_labels = self.predictions_to_yolo_labels(predictions, img_shape, conf_threshold)
        
        if not do_save or output_dir is None:
            # No writing requested; return None to indicate no file created
            if self.debug:
                print(f"[DEBUG] save_yolo_labels: do_save={do_save}, output_dir={output_dir} -> no file written")
            return None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = Path(img_path).stem
        label_path = output_dir / f"{img_name}.txt"
        
        with open(label_path, 'w') as f:
            for label in yolo_labels:
                label_str = ' '.join(map(str, label))
                f.write(label_str + '\n')
        
        return str(label_path)
    
    def process_images(self, image_paths, extract_features=True, use_gap=True):
        """
        Process multiple images and extract both predictions and losses.
        
        Args:
            image_paths: List of image paths or single path
            extract_features: If True, extract bottleneck and head features
            use_gap: If True, apply global average pooling to features
            
        Returns:
            dict: Results for each image
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        
        results = {}
        
        for img_path in image_paths:
            img_path = str(img_path)
            run_out = self.run_with_predictor(img_path, embed_layers=[-2, -1] if extract_features else None)
            results[img_path] = run_out
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = YOLOExtractor('yolo11x.pt')
    
    image_path = 'path/to/image.jpg'
    
    # Extract predictions with features (Predictor handles preprocessing + NMS)
    # Features are captured from inputs to detection head cv2 layers
    run_out = extractor.run_with_predictor(image_path, embed_layers=[-2, -1])
    first_pred = run_out['predictions'][0]
    print("Predictions:", first_pred['boxes'].shape)
    print("Gap features (cv2 inputs):", list(run_out['gap_features'].keys()))
    
    # Process multiple images with all features
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    results = extractor.process_images(image_paths, extract_features=True, use_gap=True)
    
    for img_path, data in results.items():
        print(f"\n{img_path}:")
        pred_list = data['predictions']
        print(f"  Detections: {pred_list[0]['boxes'].shape[0]}")
        print(f"  Gap features: {list(data['gap_features'].keys())}")
        print(f"  Bottleneck features: {list(data['bottleneck_features'].keys())}")
    
    # Convert predictions to YOLO labels
    img = cv2.imread(image_path)
    label_path = extractor.save_yolo_labels(
        first_pred,
        image_path,
        img_shape=img.shape,
        output_dir='labels/',
        conf_threshold=0.5,
        do_save=False  # change to True to actually write label file
    )
    
    # Use with another model
    another_model = YOLOExtractor('yolo11x.pt')
    losses = another_model.extract_losses(image_path, label_path)
    print("\nLosses:", losses)