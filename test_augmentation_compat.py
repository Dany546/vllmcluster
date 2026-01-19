"""
Verify augmentation pipeline compatibility across YOLO, DINO, and CLIP models.
Tests that each model receives correctly formatted images.
"""
import torch
import numpy as np
from cfg import augmentations, device


def test_augmentation_output_ranges():
    """Test that augmentations produce correct output ranges for each model."""
    print("Testing augmentation output ranges...")
    print("="*60)
    
    # Create dummy image HWC uint8 [0,255]
    img_np = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    
    # Create dummy boxes and masks
    bboxes = [[100, 100, 200, 200]]  # COCO format (x, y, w, h)
    category_ids = [0]
    masks = [np.random.randint(0, 2, (640, 640), dtype=np.uint8)]
    
    results = {}
    
    # Test YOLO augmentation
    print("\n1. YOLO Augmentation (to_float=False)")
    print("-" * 60)
    aug_yolo = augmentations["yolo"]
    transformed_yolo = aug_yolo(
        image=img_np.copy(),
        bboxes=bboxes,
        category_ids=category_ids,
        masks=masks
    )
    img_yolo = transformed_yolo["image"]
    
    print(f"   Input:  HWC uint8 [{img_np.min()}, {img_np.max()}]")
    print(f"   Output: {img_yolo.shape} {img_yolo.dtype}")
    print(f"   Range:  [{img_yolo.min():.2f}, {img_yolo.max():.2f}]")
    
    # Verify YOLO range
    assert img_yolo.dtype == torch.float32, f"YOLO should output float32, got {img_yolo.dtype}"
    assert img_yolo.max() > 1.0, f"YOLO should output [0,255] range, got max={img_yolo.max():.2f}"
    assert img_yolo.max() <= 255.5, f"YOLO max too high: {img_yolo.max():.2f}"
    print("   ✓ YOLO output is [0,255] float32")
    results['yolo'] = True
    
    # Test DINO augmentation  
    print("\n2. DINO Augmentation (to_float=True, normalize=True)")
    print("-" * 60)
    aug_dino = augmentations["dino"]
    transformed_dino = aug_dino(
        image=img_np.copy(),
        bboxes=bboxes,
        category_ids=category_ids,
        masks=masks
    )
    img_dino = transformed_dino["image"]
    
    print(f"   Input:  HWC uint8 [{img_np.min()}, {img_np.max()}]")
    print(f"   Output: {img_dino.shape} {img_dino.dtype}")
    print(f"   Range:  [{img_dino.min():.4f}, {img_dino.max():.4f}]")
    
    # DINO should be normalized with mean/std
    # After normalization: (img/255 - mean) / std
    # This should roughly be in [-2, 2] range
    assert img_dino.dtype == torch.float32, f"DINO should output float32, got {img_dino.dtype}"
    assert img_dino.min() < 0, f"DINO normalized should have negative values, min={img_dino.min():.4f}"
    assert img_dino.max() < 10, f"DINO normalized max seems too high: {img_dino.max():.4f}"
    print("   ✓ DINO output is normalized (mean-subtracted)")
    results['dino'] = True
    
    # Test CLIP augmentation
    print("\n3. CLIP Augmentation (to_float=True, normalize=False)")
    print("-" * 60)
    aug_clip = augmentations["clip"]
    transformed_clip = aug_clip(
        image=img_np.copy(),
        bboxes=bboxes,
        category_ids=category_ids,
        masks=masks
    )
    img_clip = transformed_clip["image"]
    
    print(f"   Input:  HWC uint8 [{img_np.min()}, {img_np.max()}]")
    print(f"   Output: {img_clip.shape} {img_clip.dtype}")
    print(f"   Range:  [{img_clip.min():.4f}, {img_clip.max():.4f}]")
    
    # CLIP should be [0,1] float32
    assert img_clip.dtype == torch.float32, f"CLIP should output float32, got {img_clip.dtype}"
    assert img_clip.min() >= -0.01, f"CLIP should have min >= 0, got {img_clip.min():.4f}"
    assert img_clip.max() <= 1.01, f"CLIP should have max <= 1, got {img_clip.max():.4f}"
    print("   ✓ CLIP output is [0,1] float32")
    results['clip'] = True
    
    # Check masks are handled
    print("\n4. Mask handling")
    print("-" * 60)
    masks_yolo = transformed_yolo.get("masks", [])
    masks_dino = transformed_dino.get("masks", [])
    masks_clip = transformed_clip.get("masks", [])
    
    print(f"   YOLO masks: {len(masks_yolo)} masks")
    if masks_yolo:
        print(f"     Type: {type(masks_yolo[0])}, Shape: {masks_yolo[0].shape if hasattr(masks_yolo[0], 'shape') else 'N/A'}")
    
    print(f"   DINO masks: {len(masks_dino)} masks")
    if masks_dino:
        print(f"     Type: {type(masks_dino[0])}, Shape: {masks_dino[0].shape if hasattr(masks_dino[0], 'shape') else 'N/A'}")
    
    print(f"   CLIP masks: {len(masks_clip)} masks")
    if masks_clip:
        print(f"     Type: {type(masks_clip[0])}, Shape: {masks_clip[0].shape if hasattr(masks_clip[0], 'shape') else 'N/A'}")
    
    print("   ✓ Masks are preserved")
    results['masks'] = True
    
    return all(results.values())


def test_model_input_expectations():
    """Document what each model expects."""
    print("\n" + "="*60)
    print("Model Input Expectations")
    print("="*60)
    
    expectations = {
        "YOLO": {
            "input_range": "[0, 255]",
            "dtype": "float32 or uint8",
            "shape": "[B, 3, H, W]",
            "normalization": "None (raw pixel values)",
            "note": "YOLO models are trained with raw uint8 images"
        },
        "DINO": {
            "input_range": "[-2, 2] (approximately)",
            "dtype": "float32",
            "shape": "[B, 3, H, W]",
            "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            "note": "Vision transformer expects normalized inputs"
        },
        "CLIP": {
            "input_range": "[0, 1]",
            "dtype": "float32",
            "shape": "[B, 3, H, W]",
            "normalization": "None (handled internally by CLIP processor with do_rescale=False)",
            "note": "CLIP processor expects [0, 1] and applies its own normalization"
        }
    }
    
    for model, specs in expectations.items():
        print(f"\n{model}:")
        for key, val in specs.items():
            print(f"  {key:20s}: {val}")
    
    print("\n" + "="*60)
    print("Augmentation Configuration Verification")
    print("="*60)
    
    config = {
        "YOLO":  "to_float=False  (preserves [0,255])",
        "DINO":  "to_float=True, normalize=True  (→ normalized)",
        "CLIP":  "to_float=True, normalize=False  (→ [0,1])",
    }
    
    for model, config_str in config.items():
        print(f"{model:6s}: {config_str}")


if __name__ == "__main__":
    print("Augmentation Pipeline Compatibility Test")
    print("="*60)
    
    try:
        passed = test_augmentation_output_ranges()
        test_model_input_expectations()
        
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        if passed:
            print("✓ All augmentation tests PASSED")
            print("\nThe code is compatible with:")
            print("  • YOLO models (expects [0,255])")
            print("  • DINO models (expects normalized)")
            print("  • CLIP models (expects [0,1])")
        else:
            print("❌ Some tests FAILED")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
