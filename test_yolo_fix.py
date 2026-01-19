"""
Quick test to verify YOLO inference is working correctly after fix.
"""
import torch
import numpy as np
from PIL import Image
import os

from cfg import augmentations, device
from clustering import yolowrapper
from dataset import COCODataset


def test_yolo_inference_with_real_data():
    """Test YOLO wrapper with real COCO data"""
    print("Testing YOLO inference with real COCO validation data...")
    
    # Load dataset
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["yolo"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    # Get first sample
    img_id, image_t, labels = dataset[0]
    
    print(f"\n1. Data Loading:")
    print(f"   Image ID: {img_id}")
    print(f"   Image shape: {image_t.shape}")
    print(f"   Image dtype: {image_t.dtype}")
    print(f"   Image range: [{image_t.min():.2f}, {image_t.max():.2f}]")
    print(f"   Number of GT boxes: {labels['cls'].shape[0]}")
    
    # Check image range
    if image_t.max() <= 1.1:
        print("   ⚠️  WARNING: Image is in [0,1] range - YOLO expects [0,255]!")
        return False
    elif image_t.max() > 200:
        print("   ✓ Image is in correct [0,255] range for YOLO")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    # Load YOLO wrapper
    print(f"\n2. Model Loading:")
    wrapper = yolowrapper("yolov8s.pt")
    print(f"   ✓ YOLO model loaded")
    
    # Get batch
    img_ids, images, labels = next(iter(dataloader))
    images = images.to(device)
    
    print(f"\n3. Batch Processing:")
    print(f"   Batch size: {images.shape[0]}")
    print(f"   Batch image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"   Total GT boxes in batch: {labels['bboxes'].shape[0]}")
    
    # Run inference
    print(f"\n4. Running Inference...")
    embeddings, outputs, all_outputs = wrapper(images, labels)
    
    print(f"\n5. Results:")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    total_predictions = 0
    for i, out in enumerate(outputs):
        miou, mconf, cat, supercat, hit_freq = out[:5]
        print(f"   Image {i}: hit_freq={hit_freq:.3f}, miou={miou:.3f}, mconf={mconf:.3f}, cat={int(cat)}")
        if hit_freq > 0:
            total_predictions += 1
    
    print(f"\n6. Summary:")
    print(f"   Images with predictions: {total_predictions}/{len(outputs)}")
    
    if total_predictions == 0:
        print("   ❌ FAILED: No predictions detected!")
        return False
    elif total_predictions < len(outputs) * 0.5:
        print(f"   ⚠️  WARNING: Only {total_predictions}/{len(outputs)} images have predictions")
        print("   This might be normal if images have no objects or difficult objects")
        return True
    else:
        print(f"   ✓ SUCCESS: {total_predictions}/{len(outputs)} images have predictions")
        return True


def test_yolo_with_original_image():
    """Test YOLO with original image (no preprocessing) to confirm model works"""
    print("\n" + "="*60)
    print("Control Test: YOLO with raw PIL image")
    print("="*60)
    
    from ultralytics import YOLO
    
    # Load dataset to get image path
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["yolo"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    img_id = dataset.ids[0]
    img_path = os.path.join(dataset.data_path, dataset.id_to_file[img_id])
    
    print(f"Loading image: {img_path}")
    orig_img = Image.open(img_path).convert("RGB")
    print(f"Image size: {orig_img.size}")
    
    # Test with raw ultralytics
    model = YOLO("yolov8s.pt")
    results = model(orig_img, verbose=False)
    
    n_preds = len(results[0].boxes)
    print(f"Raw YOLO predictions: {n_preds}")
    
    if n_preds > 0:
        print("✓ YOLO model is working correctly")
        return True
    else:
        print("⚠️  No predictions on this image (might be normal)")
        return False


if __name__ == "__main__":
    print("="*60)
    print("YOLO Inference Fix Verification")
    print("="*60)
    
    # Control test
    control_passed = test_yolo_with_original_image()
    
    # Main test
    print("\n" + "="*60)
    print("Main Test: Full Pipeline")
    print("="*60)
    main_passed = test_yolo_inference_with_real_data()
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    if main_passed:
        print("✓ YOLO inference is working correctly!")
        print("\nThe fix successfully resolved the issue:")
        print("  - Images are now kept in [0,255] range")
        print("  - YOLO model receives correctly formatted input")
        print("  - Predictions are being generated")
    else:
        print("❌ YOLO inference still has issues")
        if not control_passed:
            print("  - Even raw YOLO model isn't detecting objects")
            print("  - This might indicate model weights or image quality issues")
        else:
            print("  - Raw YOLO works, but pipeline has issues")
            print("  - Check data loading and preprocessing steps")
