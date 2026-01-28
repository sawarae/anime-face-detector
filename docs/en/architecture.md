# Architecture

This document provides a detailed explanation of the architecture and internal implementation of anime-face-detector.

## Overview

anime-face-detector consists of a two-stage detection pipeline based on the OpenMMLab ecosystem (mmdetection, mmpose).

```
Input Image → Face Detection (mmdet) → Landmark Detection (mmpose) → Result Output
```

## System Architecture

### Dependencies

This library is built on top of the OpenMMLab 2.0 framework.

```
anime-face-detector
├── mmengine (0.10.7)    # Configuration management, registry, logging
├── mmcv (2.1.0)         # Image processing, CUDA operators
├── mmdet (3.2.0)        # Object detection framework
└── mmpose (1.3.2)       # Pose estimation & landmark detection framework
```

### Directory Structure

```
anime_face_detector/
├── __init__.py              # Entry point (create_detector, get_config_path)
├── detector.py              # LandmarkDetector class (main logic)
└── configs/
    ├── mmdet/               # Face detection model configurations
    │   ├── yolov3.py        # YOLOv3 configuration
    │   └── faster-rcnn.py   # Faster R-CNN configuration
    └── mmpose/              # Landmark detection model configurations
        └── hrnetv2.py       # HRNetV2 configuration (28-point landmarks)
```

## Two-Stage Detection Pipeline

### Stage 1: Face Detection (mmdetection)

Face detection identifies bounding boxes of face regions in the image.

#### Supported Detectors

1. **YOLOv3** (Default)
   - Fast single-stage detector
   - Suitable for real-time applications
   - Model size: ~120MB

2. **Faster R-CNN**
   - High-accuracy two-stage detector
   - Recommended when detection accuracy is prioritized
   - Model size: ~160MB

#### Detection Flow

```python
# detector.py:_detect_faces()
1. Set scope with DefaultScope.overwrite_default_scope('mmdet')
2. Detect using mmdet 3.x API with inference_detector()
3. Extract bboxes and scores from DetDataSample
4. Convert to [x0, y0, x1, y1, score] format
5. Expand boxes by box_scale_factor (default 1.1x)
```

Expanding the bounding boxes ensures contextual information around the face needed for landmark detection.

### Stage 2: Landmark Detection (mmpose)

Detects 28 landmarks from the detected face regions.

#### HRNetV2 Architecture

- **Backbone**: HRNetV2 (High-Resolution Network v2)
- **Features**: Learns multi-scale features while maintaining high-resolution representations
- **Output**: 28 keypoint coordinates and confidence scores

#### Landmark Layout

The 28 landmarks cover the following regions:

```
- Face contour: 6 points (left: 0-2, right: 3-5)
- Left eyebrow: 3 points (5-7)
- Right eyebrow: 3 points (8-10)
- Left eye: 3 points (11-13)
- Right eye: 3 points (17-19)
- Nose: 3 points (14-16)
- Mouth: 7 points (23-27, 20-22)
```

#### Detection Flow

```python
# detector.py:_detect_landmarks()
1. Set scope with DefaultScope.overwrite_default_scope('mmpose')
2. Convert bbox coordinates to (N, 4) xyxy format
3. Detect using mmpose 1.x API with inference_topdown()
4. Extract keypoints and keypoint_scores from PoseDataSample
5. Convert to [x, y, score] format and return
```

## Integration with OpenMMLab 2.0

### Scope Management

Since mmdet and mmpose have their own registries, scope switching is necessary.

```python
# Detection with mmdet
with DefaultScope.overwrite_default_scope('mmdet'):
    result = inference_detector(self.face_detector, image)

# Detection with mmpose
with DefaultScope.overwrite_default_scope('mmpose'):
    results = inference_topdown(self.landmark_detector, image, bboxes)
```

### Handling API Changes

#### mmdet 3.x

- Old: `inference_detector()` returns `numpy.ndarray`
- New: `inference_detector()` returns `DetDataSample`

```python
# mmdet 3.x
result = inference_detector(model, image)
bboxes = result.pred_instances.bboxes.cpu().numpy()
scores = result.pred_instances.scores.cpu().numpy()
```

#### mmpose 1.x

- Old: Uses `inference_top_down_pose_model()`
- New: Uses `inference_topdown()`, returns `PoseDataSample`

```python
# mmpose 1.x
results = inference_topdown(model, image, bboxes, bbox_format='xyxy')
keypoints = results[0].pred_instances.keypoints[0]
scores = results[0].pred_instances.keypoint_scores[0]
```

### Configuration Management

Uses mmengine's `Config` class to load configurations.

```python
from mmengine.config import Config

config = Config.fromfile('configs/mmdet/yolov3.py')
```

Configuration files are written in Python format and support inheritance and variable definitions.

## Model Initialization

### Automatic Checkpoint Download

Model files are automatically downloaded on first run.

```python
# __init__.py:get_checkpoint_path()
model_dir = pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
# Default: ~/.cache/torch/hub/checkpoints/

if not model_path.exists():
    url = f'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/{file_name}'
    torch.hub.download_url_to_file(url, model_path.as_posix())
```

Downloaded models are reused.

### Setting dataset_meta

mmpose models require dataset-specific metadata.

```python
# detector.py:_init_pose_model()
dataset_meta = {
    'dataset_name': 'anime_face',
    'num_keypoints': 28,
    'keypoint_info': {...},
    'skeleton_info': {...},
    'joint_weights': [1.0] * 28,
    'sigmas': [0.025] * 28,
    'flip_indices': [...],
}
model.dataset_meta = dataset_meta
```

This ensures dataset information is correctly referenced during inference.

### Setting test_cfg

Dynamically applies test-time configurations like flip_test.

```python
# Enable flip_test
model.test_cfg['flip_test'] = True
model.cfg.model.test_cfg['flip_test'] = True
```

Enabling flip_test improves accuracy by also using horizontally flipped image results.

## Inference Processing Flow

### Input Processing

```python
def __call__(self, image_or_path, boxes=None):
    # 1. Load image (BGR format)
    image = self._load_image(image_or_path)

    # 2. Face detection (if boxes not specified)
    if boxes is None:
        if self.face_detector is not None:
            boxes = self._detect_faces(image)
        else:
            # Use entire image if no face detector
            h, w = image.shape[:2]
            boxes = [np.array([0, 0, w - 1, h - 1, 1])]

    # 3. Landmark detection
    return self._detect_landmarks(image, boxes)
```

### Output Format

```python
[
    {
        'bbox': np.array([x0, y0, x1, y1, score]),  # Face bounding box
        'keypoints': np.array([
            [x, y, score],  # Landmark 0
            [x, y, score],  # Landmark 1
            ...
            [x, y, score],  # Landmark 27
        ])
    },
    ...  # Repeated for each detected face
]
```

## Performance Optimization

### CUDA Optimization

mmcv can be accelerated by building with CUDA operators.

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0
```

Set `TORCH_CUDA_ARCH_LIST` according to your GPU architecture.

### Device Selection

```python
# Use CUDA
detector = create_detector('yolov3', device='cuda:0')

# Use CPU (for environments without GPU)
detector = create_detector('yolov3', device='cpu')
```

Using GPU can provide 10x or more speedup.

### Batch Processing

The current implementation processes images one at a time, but multiple images can be processed in a loop.

```python
for image_path in image_paths:
    preds = detector(image_path)
    # Process...
```

Batch inference support could be considered in the future.

## Error Handling

### Scope Errors

Registry errors occur when mmdet and mmpose scopes are mixed.

```
KeyError: 'xxx is not in the xxx registry'
```

This is avoided by properly managing scopes with `DefaultScope.overwrite_default_scope()`.

### CUDA Out of Memory

Memory exhaustion can occur when processing large or high-resolution images.

Solutions:
- Resize images
- Reduce batch size
- Use devices with larger GPU memory

### Model Download Failure

If model download fails due to network errors, manual download is possible.

```bash
cd ~/.cache/torch/hub/checkpoints/
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth
```

## Extensibility

### Adding Custom Detectors

To add a new face detector:

1. Add configuration file to `configs/mmdet/`
2. Update `get_config_path()` in `__init__.py`
3. Update `get_checkpoint_path()` in `__init__.py`
4. Update assertions in `create_detector()`

### Custom Landmark Models

When using different landmark counts or models:

1. Add configuration file to `configs/mmpose/`
2. Define landmark count in `dataset_info`
3. Prepare checkpoint file

### Custom Pre/Post-processing

You can add custom processing by inheriting `LandmarkDetector`.

```python
class CustomDetector(LandmarkDetector):
    def __call__(self, image_or_path, boxes=None):
        # Custom preprocessing
        image = self.preprocess(image_or_path)

        # Normal detection
        preds = super().__call__(image, boxes)

        # Custom postprocessing
        return self.postprocess(preds)
```

## Summary

anime-face-detector leverages the powerful capabilities of the OpenMMLab 2.0 ecosystem and achieves a highly extensible anime face detection library through modular design. The two-stage pipeline provides high-accuracy detection, and the automatic model download feature allows users to get started easily.