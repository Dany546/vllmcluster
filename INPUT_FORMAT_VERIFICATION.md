# Input Format Verification: DINO, CLIP, and YOLO

## CLIP (openai/clip-vit-base-patch16)

### Your Code
```python
# model.py, line 70
inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
```

### CLIP Input Requirements
- **Expected input range**: **[0, 1]** when `do_rescale=False`
- **Image type**: PIL Image, numpy array (HWC), or torch tensor (CHW)
- **Color space**: RGB
- **Normalization**: Applied by processor with:
  - `image_mean: [0.48145466, 0.4578275, 0.40821073]`
  - `image_std: [0.26862954, 0.26130258, 0.27577711]`

### Analysis
**✓ CORRECT in your code:**
- You use `do_rescale=False`, which means you handle rescaling yourself
- You pass images in [0, 1] range (via `to_float=True` in cfg.py for CLIP)
- The processor applies ImageNet normalization to get final [-2, 2] range

**⚠️ Critical Issue:**
- If you send [0, 255] float32 to CLIP with `do_rescale=False`, normalization breaks
- The normalization constants assume [0, 1] input
- Sending [0, 255] would result in `(128 - 0.481) / 0.268 ≈ 475` (way too large)

**Current Configuration**: ✓ **SAFE**
```python
"clip": make_albumentations_pipeline(224, normalize=False, to_float=True)  # [0,1] ✓
```

---

## DINO (DINOv2)

### Your Code
```python
# model.py, line 100
dino_features = self.dino.forward_features(image)
```

### DINO Input Requirements
- **Expected input range**: **[0, 1]** (standard for vision transformers)
- **Image type**: torch tensor CHW format
- **Shape**: [B, 3, H, W]
- **Normalization**: No internal normalization (uses raw [0, 1] values)
- **Color space**: RGB

### Analysis
**✓ CORRECT in your code:**
- You use timm ViT model which expects [0, 1] normalized inputs
- No internal rescaling or normalization
- Forward pass directly operates on [0, 1] pixel values

**⚠️ Critical Issue:**
- If you send [0, 255] to DINO, activation values will be 255x too large
- Attention weights will saturate/NaN
- Produces meaningless embeddings (no error, just wrong)

**Current Configuration**: ✓ **SAFE**
```python
"dino": make_albumentations_pipeline(518, normalize=True, to_float=True)  # [0,1] + normalization ✓
```

---

## YOLO (ultralytics)

### Your Code
```python
# clustering.py, line 449
preds = self.model(images)
```

### YOLO Input Requirements
- **Expected input range**: **[0, 255]** (uint8 format)
- **Image type**: torch tensor [B, 3, H, W]
- **Shape**: [B, 3, 640/512/etc, 640/512/etc] (model dependent)
- **Normalization**: NONE - uses raw pixel values
- **Color space**: BGR or RGB (model was trained with uint8 [0,255])

### Analysis
**✓ CORRECT in your code (after the fix):**
- YOLO models are trained with uint8 [0, 255] range
- Custom `ToTensorNoDiv` preserves [0, 255] range
- No internal rescaling - uses pixel values directly

**⚠️ Previous Issue (FIXED):**
- Original code sent [0, 1] normalized images to YOLO
- Model expects activation on pixel values, gets 255x smaller values
- Predictions collapse to near-zero → 0 detections

**Current Configuration**: ✓ **SAFE**
```python
"yolo": make_albumentations_pipeline(640, normalize=False, to_float=False)  # [0,255] ✓
```

---

## Summary Table

| Model | Input Range | Normalization | Your Config | Status |
|-------|-------------|---------------|-------------|--------|
| **CLIP** | [0, 1] | ImageNet (mean/std) | `to_float=True` | ✓ Safe |
| **DINO** | [0, 1] | ImageNet (mean/std) | `to_float=True, normalize=True` | ✓ Safe |
| **YOLO** | [0, 255] | None | `to_float=False` | ✓ Safe (after fix) |

---

## What Happens If You Get It Wrong

### Sending [0, 1] to YOLO
```
YOLO activation: 0.5 * weight
Expected: 128 * weight  (255x smaller)
Result: Predictions near zero → 0 detections ❌
```

### Sending [0, 255] to CLIP
```
CLIP normalization: (value - mean) / std
With value=128: (128 - 0.481) / 0.268 ≈ 475
Expected: (0.5 - 0.481) / 0.268 ≈ 0.07
Result: NaN or completely wrong embeddings ❌
```

### Sending [0, 255] to DINO
```
ViT attention: softmax(Q @ K^T / sqrt(d_k))
With 255x larger values in queries/keys
Result: Attention saturation → NaN or useless embeddings ❌
```

---

## Validation

Your current configuration is **CORRECT**:
- ✓ YOLO gets [0, 255] float32
- ✓ CLIP gets [0, 1] float32  + normalization applied by processor
- ✓ DINO gets [0, 1] float32 + ImageNet normalization from augmentation

The fix you implemented resolves the original YOLO zero predictions issue by preserving the [0, 255] range.
