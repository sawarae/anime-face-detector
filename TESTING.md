# Testing Guide

This document describes how to test the anime-face-detector with real models.

## Test Status

### ✅ Completed Tests

1. **Import Tests** - All modules import correctly
2. **Dependency Detection** - PyTorch and ONNX availability detection works
3. **Error Handling** - Graceful fallback when models not found
4. **Path Resolution** - ONNX model paths are correctly resolved
5. **Cache Directory** - Cache directory creation works

### ⏳ Pending Tests (Require Actual Models)

1. **Model Conversion** - PyTorch to ONNX conversion
2. **ONNX Inference** - Actual inference with ONNX models
3. **Performance Benchmarks** - Speed comparison vs PyTorch
4. **Accuracy Tests** - Output validation

## Quick Test (No Models Required)

Basic integration tests without actual model files:

```bash
# Run basic tests
python3 -c "
from anime_face_detector import create_detector
from anime_face_detector.onnx_helper import is_onnx_available
print(f'ONNX Runtime available: {is_onnx_available()}')
"
```

## Full Test (Requires Models)

### Step 1: Install PyTorch Dependencies

Choose your installation method:

**Option A: With UV (Fast)**
```bash
uv pip install -e ".[conversion]"
uv pip install openmim
uv run mim install mmcv-full mmdet mmpose
```

**Option B: With pip**
```bash
pip install openmim torch torchvision
mim install mmcv-full mmdet mmpose
```

### Step 2: Convert Models to ONNX

```bash
# Convert face detector (YOLOv3)
python tools/convert_to_onnx.py --model yolov3

# Convert landmark detector (HRNetv2)
python tools/convert_to_onnx.py --model hrnetv2
```

Expected output:
```
Converting yolov3 face detector to ONNX...
✓ Successfully exported to /root/.cache/torch/hub/checkpoints/mmdet_anime-face_yolov3.onnx
Model size: XXX.XX MB
```

### Step 3: Test ONNX Inference

```bash
# Run inference test
python test_onnx.py
```

Or test manually:

```python
import cv2
import numpy as np
from anime_face_detector import create_detector

# Create detector with ONNX backend
detector = create_detector('yolov3', device='cpu', use_onnx=True)

# Test with dummy image
dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
preds = detector(dummy_image)

print(f"Detected {len(preds)} faces")
```

### Step 4: Test with Real Images

```python
import cv2
from anime_face_detector import create_detector

# Create detector
detector = create_detector('yolov3', use_onnx=True)

# Load real image
image = cv2.imread('path/to/anime_image.jpg')

# Run detection
preds = detector(image)

for i, pred in enumerate(preds):
    print(f"Face {i}:")
    print(f"  Bounding box: {pred['bbox'][:4]}")
    print(f"  Confidence: {pred['bbox'][4]}")
    print(f"  Landmarks: {len(pred['keypoints'])} points")
```

## Performance Benchmarking

Compare ONNX vs PyTorch performance:

```python
import time
import cv2
import numpy as np
from anime_face_detector import create_detector

# Create dummy image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Test PyTorch (if available)
detector_pt = create_detector('yolov3', device='cpu', use_onnx=False)
start = time.time()
for _ in range(10):
    preds = detector_pt(image)
pt_time = (time.time() - start) / 10
print(f"PyTorch: {pt_time*1000:.2f}ms per image")

# Test ONNX
detector_onnx = create_detector('yolov3', device='cpu', use_onnx=True)
start = time.time()
for _ in range(10):
    preds = detector_onnx(image)
onnx_time = (time.time() - start) / 10
print(f"ONNX: {onnx_time*1000:.2f}ms per image")
print(f"Speedup: {pt_time/onnx_time:.2f}x")
```

## Test Results Summary

After running full tests, update this section:

| Test | Status | Notes |
|------|--------|-------|
| Import | ✅ | All modules import correctly |
| ONNX Detection | ⏳ | Requires model conversion |
| Dependency Detection | ✅ | Works correctly |
| Error Handling | ✅ | Graceful fallback |
| Model Conversion | ⏳ | Pending |
| Inference Speed | ⏳ | Pending |
| Accuracy | ⏳ | Pending |

## Known Issues

None identified in basic tests. Full model testing pending.

## Next Steps

1. Set up CI/CD with pre-converted ONNX models
2. Add automated performance benchmarks
3. Create unit tests for ONNX helper functions
4. Add regression tests for model accuracy
