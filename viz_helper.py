# viz_helpers.py
import os
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def _to_uint8_image(img):
    """
    Convert HWC image (numpy or torch) to uint8 HWC in range [0,255].
    Accepts float in [0,1], float normalized, or uint8.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().permute(1, 2, 0).numpy()
    img = np.asarray(img)
    if img.dtype in (np.float32, np.float64):
        # if values look normalized to [0,1]
        if img.max() <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).round().astype(np.uint8)
    elif img.dtype == np.uint8:
        pass
    else:
        img = img.astype(np.uint8)
    # ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    return img

def _draw_coco_bbox(ax, bbox, img_h, img_w, color="lime", label=None):
    """
    bbox: either normalized YOLO [xc,yc,w,h] in [0,1] or COCO pixel [x,y,w,h]
    """
    x, y, w, h = bbox
    # heuristic: normalized if all <= 1.0
    if max(x, y, w, h) <= 1.0:
        xc = x * img_w
        yc = y * img_h
        bw = w * img_w
        bh = h * img_h
        x_min = xc - bw / 2.0
        y_min = yc - bh / 2.0
    else:
        x_min = x
        y_min = y
        bw = w
        bh = h
    rect = patches.Rectangle((x_min, y_min), bw, bh, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    if label is not None:
        ax.text(x_min, max(0, y_min - 6), str(label), color="white", fontsize=9,
                bbox=dict(facecolor=color, alpha=0.7, pad=0))

def _draw_polygon(ax, polygon, img_h, img_w, color="yellow", normalized=False):
    pts = np.array(polygon).reshape(-1, 2)
    if normalized:
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
    poly = patches.Polygon(pts, closed=True, linewidth=1.5, edgecolor=color, facecolor="none")
    ax.add_patch(poly)

def _overlay_mask(ax, mask_np, img_h, img_w, color=(1.0, 0.0, 0.0, 0.35)):
    """
    mask_np: 2D numpy array (H_mask, W_mask) with 0/1 values
    Resizes mask to img_h x img_w using PIL.NEAREST if needed.
    """
    if mask_np.shape != (img_h, img_w):
        pil = Image.fromarray((mask_np > 0).astype("uint8") * 255).convert("L")
        pil = pil.resize((img_w, img_h), resample=Image.NEAREST)
        mask_np = (np.array(pil) > 127).astype(np.uint8)
    colored = np.zeros((img_h, img_w, 4), dtype=np.float32)
    colored[..., 0] = color[0]
    colored[..., 1] = color[1] if len(color) > 1 else 0.0
    colored[..., 2] = color[2] if len(color) > 2 else 0.0
    colored[..., 3] = color[3] if len(color) > 3 else 0.35
    colored[..., 3] *= (mask_np > 0).astype(np.float32)
    ax.imshow(colored)

def save_first_image_with_labels(
    batch,
    save_name: str = "debug_first_image.png",
    images_dir: Optional[str] = None,
    id_to_file: Optional[Dict[int, str]] = None,
    original_annotations: Optional[Dict[int, List[dict]]] = None,
    show_original_labels: bool = True,
):
    """
    Save a side-by-side figure: original image (with original COCO labels if provided)
    and transformed image (first image of first batch) with dataset labels.
    File is saved in the current working directory under save_name.
    """
    ids, images, labels = batch
    # pick first image in batch
    img_t = images[0].cpu()
    if img_t.ndim != 3:
        raise ValueError("images tensor must be (B,C,H,W)")
    C, H, W = img_t.shape

    trans_vis = _to_uint8_image(img_t)  # HWC uint8

    # load original image if possible
    orig_vis = None
    orig_id = ids[0] if isinstance(ids, (list, tuple)) and len(ids) > 0 else None
    if images_dir and id_to_file and orig_id is not None:
        candidate = os.path.join(images_dir, id_to_file[orig_id])
        if not os.path.exists(candidate):
            # try common extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                p = candidate + ext
                if os.path.exists(p):
                    candidate = p
                    break
        if os.path.exists(candidate):
            orig = Image.open(candidate).convert("RGB")
            orig_vis = np.array(orig).astype(np.uint8)

    # layout
    ncols = 2 if orig_vis is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    # original panel
    if orig_vis is not None:
        ax0 = axes[0]
        ax0.imshow(orig_vis)
        ax0.axis("off")
        ax0.set_title("Original image")
        if show_original_labels and original_annotations:
            for ann in original_annotations[orig_id]:
                if "bbox" in ann:
                    # COCO bbox in pixels
                    x, y, w, h = ann["bbox"]
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="cyan", facecolor="none")
                    ax0.add_patch(rect)
                if "segmentation" in ann and ann["segmentation"]:
                    seg = ann["segmentation"]
                    if isinstance(seg, list):
                        for poly in seg:
                            _draw_polygon(ax0, poly, orig_vis.shape[0], orig_vis.shape[1], color="yellow", normalized=False)

    # transformed panel
    ax_t = axes[-1]
    ax_t.imshow(trans_vis)
    ax_t.axis("off")
    ax_t.set_title("Transformed image with labels")

    # determine instance indices for the first image
    if "batch_idx" in labels:
        batch_idx = labels["batch_idx"].cpu()
        inst_idx = torch.where(batch_idx == 0)[0]
    else:
        inst_idx = torch.arange(labels["cls"].shape[0]) if "cls" in labels else torch.tensor([], dtype=torch.long)

    # draw transformed bboxes
    if "bboxes" in labels and inst_idx.numel() > 0:
        bboxes = labels["bboxes"].cpu()
        classes = labels.get("cls", None)
        for i in inst_idx:
            idx = int(i)
            bbox = bboxes[idx].numpy()
            cls_label = None
            if classes is not None and classes.ndim >= 2 and classes.shape[1] >= 1:
                cls_label = int(classes[idx][0].item())
            _draw_coco_bbox(ax_t, bbox, H, W, color="lime", label=cls_label)

    # overlay transformed masks if present
    if "masks" in labels and labels["masks"].numel() > 0 and inst_idx.numel() > 0:
        masks = labels["masks"].cpu()
        for i in inst_idx:
            idx = int(i)
            mask = masks[idx].numpy()
            _overlay_mask(ax_t, mask, H, W, color=(1.0, 0.0, 0.0, 0.35))

    # save to current working directory
    save_path = os.path.abspath(os.path.join(os.getcwd(), save_name))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("Real image integration test passed. Visualization saved at:", save_path)
    return save_path
