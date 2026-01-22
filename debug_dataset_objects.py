"""
Debug script to check if COCO validation data has actual objects.
"""
import sys
sys.path.insert(0, '/home/ucl/irec/darimez/MIRO/vllmcluster')

from cfg import augmentations
from dataset import COCODataset

# Load dataset
dataset = COCODataset(
    data_split="validation",
    transform=augmentations["yolo"],
    caching=False,
    get_features=False,
    segmentation=False,
)

print(f"Dataset size: {len(dataset)}")
print("\n" + "="*60)
print("Checking first 10 samples for objects")
print("="*60)

empty_count = 0
with_objects = 0

for i in range(min(10, len(dataset))):
    img_id, image_t, labels = dataset[i]
    n_boxes = labels['bboxes'].shape[0]
    
    print(f"\nSample {i} (img_id={img_id}):")
    print(f"  Image range: [{image_t.min():.4f}, {image_t.max():.4f}]")
    print(f"  Number of boxes: {n_boxes}")
    
    if n_boxes > 0:
        print(f"  Classes: {labels['cls'].squeeze().numpy()}")
        print(f"  Bboxes (first 3):\n{labels['bboxes'][:min(3)]}")
        with_objects += 1
    else:
        print(f"  ⚠️  NO OBJECTS IN THIS IMAGE")
        empty_count += 1

print("\n" + "="*60)
print(f"Summary: {with_objects} images with objects, {empty_count} empty images")
print("="*60)

if empty_count >= 7:
    print("⚠️  WARNING: Mostly empty validation set!")
    print("This would explain 'no detections'")
elif with_objects > 0:
    print("✓ Validation set has objects")
    print("If YOLO still detects nothing, issue is in model/preprocessing")
