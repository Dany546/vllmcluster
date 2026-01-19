"""
Integration test: Verify actual models work with their augmentations.
This tests the full pipeline: data → augmentation → model forward pass.
"""
import torch
import numpy as np
from cfg import augmentations, device
from dataset import COCODataset


def test_yolo_inference():
    """Test YOLO model with YOLO augmentation"""
    print("\n" + "="*60)
    print("TEST: YOLO Model with YOLO Augmentation")
    print("="*60)
    
    from clustering import yolowrapper
    
    # Load dataset with YOLO augmentation
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["yolo"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    # Get a sample
    img_id, image_t, labels = dataset[0]
    
    print(f"Image range: [{image_t.min():.2f}, {image_t.max():.2f}]")
    assert image_t.max() > 100, f"YOLO image should be in [0,255], got max={image_t.max():.2f}"
    print("✓ Image is in [0, 255] range")
    
    # Create batch
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=dataset.collate_fn, num_workers=0
    )
    img_ids, images, labels_batch = next(iter(dataloader))
    images = images.to(device)
    
    # Run model
    wrapper = yolowrapper("yolov8s.pt")
    embeddings, outputs, all_outputs = wrapper(images, labels_batch)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Outputs: {len(outputs)} images")
    print("✓ YOLO forward pass successful")
    
    return True


def test_dino_inference():
    """Test DINO model with DINO augmentation"""
    print("\n" + "="*60)
    print("TEST: DINO Model with DINO Augmentation")
    print("="*60)
    
    from model import DINO
    
    # Load dataset with DINO augmentation
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["dino"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    # Get a sample
    img_id, image_t, labels = dataset[0]
    
    print(f"Image range: [{image_t.min():.4f}, {image_t.max():.4f}]")
    # After normalization, should be roughly [-2, 2]
    assert image_t.min() < 0, f"DINO image should be normalized (have negative values), min={image_t.min():.4f}"
    print("✓ Image is normalized")
    
    # Create batch
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=dataset.collate_fn, num_workers=0
    )
    img_ids, images, labels_batch = next(iter(dataloader))
    images = images.to(device)
    
    # Run model
    dino = DINO(attention_pooling=False).to(device)
    embeddings = dino(images)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("✓ DINO forward pass successful")
    
    return True


def test_clip_inference():
    """Test CLIP model with CLIP augmentation"""
    print("\n" + "="*60)
    print("TEST: CLIP Model with CLIP Augmentation")
    print("="*60)
    
    from model import CLIP
    
    # Load dataset with CLIP augmentation
    dataset = COCODataset(
        data_split="validation",
        transform=augmentations["clip"],
        caching=False,
        get_features=False,
        segmentation=False,
    )
    
    # Get a sample
    img_id, image_t, labels = dataset[0]
    
    print(f"Image range: [{image_t.min():.4f}, {image_t.max():.4f}]")
    assert 0 <= image_t.max() <= 1.01, f"CLIP image should be in [0,1], got max={image_t.max():.4f}"
    print("✓ Image is in [0, 1] range")
    
    # Create batch
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=dataset.collate_fn, num_workers=0
    )
    img_ids, images, labels_batch = next(iter(dataloader))
    images = images.to(device)
    
    # Run model
    clip = CLIP().to(device)
    embeddings = clip(images)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("✓ CLIP forward pass successful")
    
    return True


if __name__ == "__main__":
    print("Full Integration Test: Models with Correct Augmentations")
    print("="*60)
    
    results = {}
    
    try:
        print("\nTesting YOLO...")
        results['yolo'] = test_yolo_inference()
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        import traceback
        traceback.print_exc()
        results['yolo'] = False
    
    try:
        print("\nTesting DINO...")
        results['dino'] = test_dino_inference()
    except Exception as e:
        print(f"❌ DINO test failed: {e}")
        import traceback
        traceback.print_exc()
        results['dino'] = False
    
    try:
        print("\nTesting CLIP...")
        results['clip'] = test_clip_inference()
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        results['clip'] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {model.upper()}: {'PASSED' if passed else 'FAILED'}")
    
    if all(results.values()):
        print("\n✓ All models compatible with their augmentations!")
    else:
        print("\n❌ Some tests failed - check compatibility")
