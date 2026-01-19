"""
Debug script to identify YOLO inference issues.
Tests both model creation and data loading.
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from cfg import augmentations, device
from clustering import yolowrapper
from dataset import COCODataset


def test_1_raw_yolo_model():
    """Test 1: Load YOLO model directly with raw ultralytics API"""
    print("\n" + "="*60)
    print("TEST 1: Raw YOLO model with dummy image")
    print("="*60)
    
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    
    # Create a dummy 640x640 RGB image with some patterns
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Add some high-contrast patterns to ensure detections
    dummy_img[100:200, 100:200, :] = [255, 0, 0]  # red box
    dummy_img[300:400, 300:400, :] = [0, 255, 0]  # green box
    
    results = model(dummy_img, save=False, verbose=False)
    
    print(f"Number of predictions: {len(results[0].boxes)}")
    if len(results[0].boxes) > 0:
        print(f"Confidences: {results[0].boxes.conf.cpu().numpy()}")
        print(f"Classes: {results[0].boxes.cls.cpu().numpy()}")
    else:
        print("⚠️  WARNING: No predictions from raw YOLO model!")
    
    return len(results[0].boxes) > 0


def test_2_yolo_wrapper():
    """Test 2: Test yolowrapper with dummy tensor"""
    print("\n" + "="*60)
    print("TEST 2: YOLOWrapper with dummy tensor")
    print("="*60)
    
    wrapper = yolowrapper("yolov8s.pt")
    
    # Create dummy batch [B, C, H, W] with values in [0, 255] range
    dummy_tensor = torch.randint(0, 255, (2, 3, 640, 640), dtype=torch.float32).to(device)
    
    # Add high-contrast patterns
    dummy_tensor[0, :, 100:200, 100:200] = 255.0
    dummy_tensor[1, :, 300:400, 300:400] = 255.0
    
    print(f"Input tensor shape: {dummy_tensor.shape}")
    print(f"Input tensor range: [{dummy_tensor.min():.2f}, {dummy_tensor.max():.2f}]")
    print(f"Input tensor dtype: {dummy_tensor.dtype}")
    
    # Create dummy targets
    dummy_targets = {
        'bboxes': torch.tensor([[0.25, 0.25, 0.35, 0.35]], dtype=torch.float32).to(device),
        'cls': torch.tensor([[0]], dtype=torch.long).to(device),
        'batch_idx': torch.tensor([0], dtype=torch.long).to(device),
    }
    
    embeddings, outputs, all_outputs = wrapper(dummy_tensor, dummy_targets)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Image {i}: {out}")
    
    return any(out[4] > 0 for out in outputs)  # hit_freq > 0


def test_3_data_loading():
    """Test 3: Load real data and check preprocessing"""
    print("\n" + "="*60)
    print("TEST 3: Data loading and preprocessing")
    print("="*60)
    
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["yolo"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    # Get first sample
    img_id, image_t, labels = dataset[0]
    
    print(f"Image ID: {img_id}")
    print(f"Image tensor shape: {image_t.shape}")
    print(f"Image tensor dtype: {image_t.dtype}")
    print(f"Image tensor range: [{image_t.min():.2f}, {image_t.max():.2f}]")
    print(f"Number of annotations: {labels['cls'].shape[0]}")
    print(f"Classes: {labels['cls'].squeeze().numpy()}")
    print(f"Bboxes shape: {labels['bboxes'].shape}")
    if labels['bboxes'].shape[0] > 0:
        print(f"Bboxes (first 3):\n{labels['bboxes'][:3]}")
        print(f"Bbox range: [{labels['bboxes'].min():.4f}, {labels['bboxes'].max():.4f}]")
    
    # Check if image is normalized
    if image_t.max() <= 1.0:
        print("⚠️  WARNING: Image appears to be normalized to [0,1]")
        print("   YOLO expects pixel values in [0, 255] range!")
    elif image_t.max() > 1.0 and image_t.max() <= 255:
        print("✓ Image is in correct range [0, 255]")
    else:
        print(f"⚠️  WARNING: Unexpected image range!")
    
    return image_t


def test_4_full_pipeline():
    """Test 4: Full pipeline with real data"""
    print("\n" + "="*60)
    print("TEST 4: Full pipeline with real COCO data")
    print("="*60)
    
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["yolo"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    wrapper = yolowrapper("yolov8s.pt")
    
    # Get first batch
    img_ids, images, labels = next(iter(dataloader))
    
    print(f"Batch images shape: {images.shape}")
    print(f"Images range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Labels bboxes shape: {labels['bboxes'].shape}")
    print(f"Labels cls shape: {labels['cls'].shape}")
    
    # Run inference
    images = images.to(device)
    embeddings, outputs, all_outputs = wrapper(images, labels)
    
    print(f"\nResults:")
    print(f"Embeddings shape: {embeddings.shape}")
    for i, out in enumerate(outputs):
        miou, mconf, cat, supercat, hit_freq = out[:5]
        print(f"  Image {i}: hit_freq={hit_freq:.2f}, miou={miou:.2f}, mconf={mconf:.2f}")
    
    total_hits = sum(out[4] for out in outputs)
    print(f"\nTotal hit frequency: {total_hits:.2f}")
    
    return total_hits > 0


def test_5_image_normalization():
    """Test 5: Check if normalization is the issue"""
    print("\n" + "="*60)
    print("TEST 5: Test with unnormalized vs normalized images")
    print("="*60)
    
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    
    # Load a real image
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["yolo"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    img_id, image_t, labels = dataset[0]
    
    # Test 1: As-is from dataset
    img_numpy = image_t.permute(1, 2, 0).numpy()
    results1 = model(img_numpy, save=False, verbose=False)
    pred1 = len(results1[0].boxes)
    print(f"Predictions with dataset image (range {img_numpy.min():.2f}-{img_numpy.max():.2f}): {pred1}")
    
    # Test 2: Ensure [0-255] range
    if img_numpy.max() <= 1.0:
        img_numpy_scaled = (img_numpy * 255).astype(np.uint8)
    else:
        img_numpy_scaled = img_numpy.astype(np.uint8)
    
    results2 = model(img_numpy_scaled, save=False, verbose=False)
    pred2 = len(results2[0].boxes)
    print(f"Predictions with [0-255] image: {pred2}")
    
    # Load original image without augmentation
    from PIL import Image
    import os
    img_path = os.path.join(dataset.data_path, dataset.id_to_file[img_id])
    orig_img = Image.open(img_path).convert("RGB")
    results3 = model(orig_img, verbose=False)
    pred3 = len(results3[0].boxes)
    print(f"Predictions with raw PIL image: {pred3}")
    
    return pred1, pred2, pred3


if __name__ == "__main__":
    print("YOLO Inference Debugging Script")
    print("="*60)
    
    results = {}
    
    try:
        results['test1'] = test_1_raw_yolo_model()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results['test1'] = False
    
    try:
        results['test2'] = test_2_yolo_wrapper()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results['test2'] = False
    
    try:
        image_t = test_3_data_loading()
        results['test3'] = True
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results['test3'] = False
        image_t = None
    
    try:
        results['test4'] = test_4_full_pipeline()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        results['test4'] = False
    
    try:
        pred1, pred2, pred3 = test_5_image_normalization()
        results['test5'] = (pred1, pred2, pred3)
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        import traceback
        traceback.print_exc()
        results['test5'] = None
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, result in results.items():
        status = "✓" if result else "❌"
        print(f"{status} {test}: {result}")
    
    # Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if not results.get('test1'):
        print("❌ Raw YOLO model failed - model file may be corrupted")
    elif not results.get('test2'):
        print("❌ YOLOWrapper has issues - check wrapper implementation")
    elif not results.get('test4'):
        print("❌ Full pipeline failed - likely data preprocessing issue")
        if image_t is not None and image_t.max() <= 1.0:
            print("   → LIKELY CAUSE: Images are normalized to [0,1]")
            print("   → SOLUTION: YOLO expects [0,255] range, disable normalization")
    else:
        print("✓ All tests passed!")
