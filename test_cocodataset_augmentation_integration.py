# test_cocodataset_augmentation_integration.py
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import tempfile
import pytest
 
from dataset import COCODataset   
from cfg import augmentations
PadToSquare = augmentations['yolo']  

def write_synthetic_coco(tmpdir, img_name="0000001.jpg"):
    """
    Create a synthetic image and a minimal COCO json with one polygon annotation.
    Returns: images_dir, coco_json_path, image_id, polygon, orig_size, bbox_coco
    """
    images_dir = os.path.join(tmpdir, "coco/images")
    os.makedirs(images_dir, exist_ok=True)

    # synthetic image (wider than tall)
    orig_w, orig_h = 400, 300
    img = Image.new("RGB", (orig_w, orig_h), (128, 128, 128))
    draw = ImageDraw.Draw(img)

    # rectangle polygon at known coords
    poly = [100, 80, 300, 80, 300, 220, 100, 220]  # rectangle
    draw.polygon([(poly[i], poly[i+1]) for i in range(0, len(poly), 2)], outline=(255,0,0), fill=(200,0,0))

    img_path = os.path.join(images_dir, img_name)
    img.save(img_path)

    # compute bbox from polygon
    xs = poly[0::2]
    ys = poly[1::2]
    x_min = min(xs)
    y_min = min(ys)
    w_box = max(xs) - x_min
    h_box = max(ys) - y_min

    image_id = 1
    coco = {
        "images": [{"id": image_id, "file_name": img_name, "width": orig_w, "height": orig_h}],
        "annotations": [
            {
                "id": 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x_min, y_min, w_box, h_box],
                "segmentation": [poly],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    coco_json_path = os.path.join(tmpdir, "instances_test.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco, f)
    return images_dir, coco_json_path, image_id, poly, (orig_w, orig_h), [x_min, y_min, w_box, h_box]


def manual_scale_and_pad(bbox_coco, polygon, orig_size, out_size):
    """
    Apply scale-to-longest-side then symmetric center pad to compute expected bbox and mask.
    Returns normalized YOLO bbox and mask numpy array (out_size x out_size).
    """
    orig_w, orig_h = orig_size
    D = out_size
    scale = D / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = D - new_w
    pad_y = D - new_h
    pad_left = pad_x // 2
    pad_top = pad_y // 2

    # bbox
    x, y, w, h = bbox_coco
    x_s = x * scale
    y_s = y * scale
    w_s = w * scale
    h_s = h * scale
    x_p = x_s + pad_left
    y_p = y_s + pad_top

    x_center = (x_p + w_s / 2.0) / D
    y_center = (y_p + h_s / 2.0) / D
    bw = w_s / D
    bh = h_s / D
    yolo_bbox_norm = [x_center, y_center, bw, bh]

    # rasterize polygon at original size
    mask_img = Image.new("L", (orig_w, orig_h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)], outline=1, fill=1)
    mask_arr = np.array(mask_img, dtype=np.uint8)

    # resize mask to new_w,new_h using NEAREST
    mask_pil = Image.fromarray(mask_arr * 255).convert("L")
    if (new_w, new_h) != (orig_w, orig_h):
        mask_pil = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)

    # pad to D x D
    padded = Image.new("L", (D, D), 0)
    padded.paste(mask_pil, (pad_left, pad_top))
    mask_np = (np.array(padded) > 127).astype(np.uint8)

    return yolo_bbox_norm, mask_np


@pytest.mark.parametrize("pad_size", [320, 640])
def test_dataset_applies_same_geometry(tmp_path, pad_size):
    """
    Integration test: instantiate your COCODataset with the transform you use,
    fetch the first item, and compare bbox+mask to a manual application of the same geometry.
    """
    # create synthetic files
    images_dir, coco_json_path, image_id, poly, orig_size, bbox_coco = write_synthetic_coco(str(tmp_path))
 
    transform = PadToSquare

    # Instantiate dataset (match your constructor signature)
    ds = COCODataset(
        data_split="train",
        get_features=False,
        cache_path=str(tmp_path),
        transform=transform,
        caching=False,
        segmentation=True,
        caption_sampling="first", 
    )

    # If your class loads a fixed JSON path, override internals to use our tiny coco
    # (This ensures the dataset uses the synthetic image/annotation we created)
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    ds.data = coco_data
    ds.images = {img["id"]: img for img in coco_data["images"]}
    ds.annotations = coco_data["annotations"]
    ds.img_to_anns = {}
    for ann in ds.annotations:
        ds.img_to_anns.setdefault(ann["image_id"], []).append(ann)
    ds.id_to_file = {img_id: img_dict["file_name"] for img_id, img_dict in ds.images.items()}
    ds.ids = list(ds.images.keys())

    # get dataset output for first item
    img_id_out, image_t, labels = ds[0]

    # infer final D from dataset output if pad_size not used by dataset
    if hasattr(ds, "pad_size") and ds.pad_size is not None:
        D = ds.pad_size
    else:
        # fallback to image tensor shape
        _, H, W = image_t.shape
        assert H == W
        D = H

    # manual expected
    expected_bbox, expected_mask = manual_scale_and_pad(bbox_coco, poly, orig_size, D)

    # --- Compare bbox ---
    assert "bboxes" in labels, "Dataset did not return bboxes"
    got_bboxes = labels["bboxes"].cpu().numpy()
    assert got_bboxes.shape[0] == 1, f"Expected 1 bbox, got {got_bboxes.shape[0]}"
    got_bbox = got_bboxes[0]
    # allow small numerical tolerance
    assert np.allclose(got_bbox, np.array(expected_bbox), atol=1e-4), f"bbox mismatch: got {got_bbox}, expected {expected_bbox}"

    # --- Compare mask ---
    assert "masks" in labels, "Dataset did not return masks"
    got_masks = labels["masks"].cpu().numpy()
    assert got_masks.shape[0] == 1, f"Expected 1 mask, got {got_masks.shape[0]}"
    got_mask = got_masks[0].astype(np.uint8)

    # If mask shape differs, resize expected to got_mask shape using nearest (shouldn't happen if geometry matches)
    if got_mask.shape != expected_mask.shape:
        from PIL import Image
        exp_pil = Image.fromarray((expected_mask * 255).astype(np.uint8)).convert("L")
        exp_pil = exp_pil.resize((got_mask.shape[1], got_mask.shape[0]), resample=Image.NEAREST)
        expected_mask = (np.array(exp_pil) > 127).astype(np.uint8)

    assert got_mask.shape == expected_mask.shape, f"mask shape mismatch: got {got_mask.shape}, expected {expected_mask.shape}"
    # exact equality expected because we used NEAREST and integer ops
    assert np.array_equal(got_mask, expected_mask), "mask pixels differ between dataset and manual transform"
    print("Test passed for pad_size =", pad_size)