"""Quick test to verify the mask tensor conversion fix"""
import torch
import numpy as np

# Simulate what happens in dataset.py
def test_mask_conversion():
    print("Testing mask conversion from numpy arrays...")
    
    # Simulate masks coming from Albumentations (as numpy arrays)
    masks = [
        np.random.randint(0, 2, (640, 640), dtype=np.uint8),
        np.random.randint(0, 2, (640, 640), dtype=np.uint8),
    ]
    
    print(f"Input: {len(masks)} masks as numpy arrays")
    print(f"  Type: {type(masks[0])}")
    print(f"  Shape: {masks[0].shape}")
    
    # Apply the fix
    mask_tensors = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            mask_tensors.append(m.to(torch.uint8))
        else:
            # numpy array
            mask_tensors.append(torch.from_numpy(m).to(torch.uint8))
    
    masks_t = torch.stack(mask_tensors, dim=0)
    
    print(f"\nOutput: Stacked tensor")
    print(f"  Type: {type(masks_t)}")
    print(f"  Shape: {masks_t.shape}")
    print(f"  Dtype: {masks_t.dtype}")
    
    assert isinstance(masks_t, torch.Tensor), "Should be tensor"
    assert masks_t.shape == (2, 640, 640), f"Wrong shape: {masks_t.shape}"
    assert masks_t.dtype == torch.uint8, f"Wrong dtype: {masks_t.dtype}"
    
    print("\n✓ Test passed!")
    return True

if __name__ == "__main__":
    test_mask_conversion()
