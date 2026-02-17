# Models and Configuration Files

This document provides detailed explanations of the models and configuration files used in anime-face-detector.

## Model Overview

anime-face-detector provides three pre-trained models:

| Model | Task | Architecture | Size | Use Case |
|-------|------|--------------|------|----------|
| YOLOv3 | Face Detection | Darknet-53 | ~120MB | Fast detection |
| Faster R-CNN | Face Detection | ResNet-50 + FPN | ~160MB | High accuracy detection |
| HRNetV2 | Landmark Detection | HRNetV2 | ~100MB | 28-point landmarks |

## Model Downloads

Models are automatically downloaded from GitHub Releases on first run.

- **Download source**: https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1
- **Save location**: `~/.cache/torch/hub/checkpoints/`

Manual download is also possible:

```bash
cd ~/.cache/torch/hub/checkpoints/
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_faster-rcnn.pth
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth
```

---

## Face Detection Models

### YOLOv3

#### Architecture

- **Backbone**: Darknet-53 (53-layer convolutional network)
- **Neck**: YOLOV3Neck (3-scale FPN)
- **Head**: YOLOV3Head (1-class detection)

#### Configuration File: configs/mmdet/yolov3.py

```python
model = dict(
    type='YOLOV3',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # Normalize images to 0-1
        bgr_to_rgb=True,
        pad_size_divisor=32,        # Pad to multiples of 32
    ),
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5)       # Output feature maps at 3 scales
    ),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128],
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=1,               # Anime faces only
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],  # Large faces
                [(30, 61), (62, 45), (59, 119)],      # Medium faces
                [(10, 13), (16, 30), (33, 23)],       # Small faces
            ],
            strides=[32, 16, 8],
        ),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
    ),
    test_cfg=dict(
        nms_pre=1000,               # Max boxes before NMS
        min_bbox_size=0,
        score_thr=0.05,             # Score threshold
        conf_thr=0.005,             # Confidence threshold
        nms=dict(type='nms', iou_threshold=0.45),  # NMS IoU threshold
        max_per_img=100,            # Max detections per image
    ),
)
```

#### Test Pipeline

```python
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),  # Resize while maintaining aspect ratio
    dict(type='Pad', size=(608, 608), pad_val=dict(img=(114, 114, 114))),  # Pad with gray
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
```

#### Features

- **Speed**: Very fast (GPU: 30-50 FPS)
- **Accuracy**: Good (mAP: ~0.95)
- **Memory usage**: Moderate
- **Recommended use**: Real-time applications, video processing

---

### Faster R-CNN

#### Architecture

- **Backbone**: ResNet-50
- **Neck**: FPN (Feature Pyramid Network)
- **RPN**: Region Proposal Network
- **Head**: Shared2FCBBoxHead (two-stage detection)

#### Configuration File: configs/mmdet/faster-rcnn.py

Key components (differences from YOLOv3):

```python
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,             # Freeze stage1
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,           # Anime faces only
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)
```

#### Features

- **Speed**: Medium (GPU: 10-20 FPS)
- **Accuracy**: High accuracy (mAP: ~0.97)
- **Memory usage**: Larger
- **Recommended use**: Accuracy-focused applications, still image processing

---

## Landmark Detection Model

### HRNetV2

#### Architecture

High-Resolution Network v2 learns multi-scale features while maintaining high-resolution representations.

```
Input (256x256)
  ↓
Stage 1: 64 channels (1 branch)
  ↓
Stage 2: 18, 36 channels (2 branches)
  ↓
Stage 3: 18, 36, 72 channels (3 branches) ← Repeated 4 times
  ↓
Stage 4: 18, 36, 72, 144 channels (4 branches) ← Repeated 3 times
  ↓
Concat: 270 channels (18+36+72+144)
  ↓
Head: 28-channel heatmap
```

#### Configuration File: configs/mmpose/hrnetv2.py

```python
# Codec configuration (coordinate ⇄ heatmap conversion)
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),       # Input size
    heatmap_size=(64, 64),       # Heatmap size (1/4)
    sigma=2,                     # Gaussian kernel standard deviation
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet statistics
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True,  # Output all branches
            ),
        ),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,                     # Concatenate all branches
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=270,                 # 18+36+72+144
        out_channels=28,                 # 28-point landmarks
        deconv_out_channels=None,
        conv_out_channels=(270,),
        conv_kernel_sizes=(1,),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,                 # flip_test controlled by LandmarkDetector
    ),
)
```

#### Flip Indices (Left-Right Flip Mapping)

Indices for swapping symmetric landmarks during horizontal flipping:

```python
flip_indices = [
    4, 3, 2, 1, 0,        # Face contour (left-right flip)
    10, 9, 8, 7, 6, 5,    # Eyebrows (left-right flip)
    19, 18, 17,           # Right eye ↔ Left eye
    22, 21, 20,           # Nose/mouth (left-right flip)
    13, 12, 11,           # Left eye ↔ Right eye
    16, 15, 14,           # Mouth (left-right flip)
    23,                   # Center (unchanged)
    26, 25, 24,           # Lower mouth (left-right flip)
    27                    # Center (unchanged)
]
```

#### Dataset Metainfo

Defines landmark metadata:

```python
dataset_info = dict(
    dataset_name='anime_face',
    keypoint_info={
        0: dict(name='kpt-0', id=0, color=[255, 255, 255], swap='kpt-4'),
        # ... information for each landmark
        27: dict(name='kpt-27', id=27, color=[255, 255, 255], swap=''),
    },
    skeleton_info={},           # Skeleton connections (unused)
    joint_weights=[1.0] * 28,   # Weight for each landmark (all equal)
    sigmas=[0.025] * 28,        # Standard deviation for OKS calculation
    flip_indices=flip_indices,
)
```

#### Test Pipeline

```python
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),              # Calculate box center and scale
    dict(type='TopdownAffine', input_size=(256, 256)),  # Normalize with affine transform
    dict(type='PackPoseInputs'),
]
```

#### Features

- **Accuracy**: Very high accuracy (PCK@0.1: ~0.95)
- **Speed**: Medium (GPU: 50-100 FPS)
- **Memory usage**: Moderate
- **28-point landmarks**: Detailed landmarks specifically for anime faces

---

## Landmark Layout Details

### 28-Point Layout

```
         5   6   7                 8   9  10
           ●   ●   ●           ●   ●   ●
          (Left eyebrow)        (Right eyebrow)

    11  12  13                   17  18  19
      ●   ●   ●                   ●   ●   ●
       (Left eye)                 (Right eye)

              14  15  16
                ●   ●   ●
                 (Nose)

         20  21  22
          ●   ●   ●
           (Mouth right)

              23
               ●
            (Mouth center bottom)

         24  25  26
          ●   ●   ●
           (Lower mouth)

              27
               ●
           (Mouth center)

0   1   2                         3   4
 ●   ●   ●                         ●   ●
 (Left contour)                   (Right contour)
```

### Index and Region Mapping

| Index | Region | Description |
|-------|--------|-------------|
| 0-2 | Left contour | Left side face contour |
| 3-5 | Right contour | Right side face contour |
| 5-7 | Left eyebrow | 3 points of left eyebrow |
| 8-10 | Right eyebrow | 3 points of right eyebrow |
| 11-13 | Left eye | 3 points of left eye |
| 14-16 | Nose | 3 points of nose |
| 17-19 | Right eye | 3 points of right eye |
| 20-22 | Mouth right | 3 points on right side of mouth |
| 23 | Mouth center bottom | Center bottom of mouth |
| 24-26 | Lower mouth | 3 points of lower mouth |
| 27 | Mouth center | Center of mouth |

---

## Customizing Configuration Files

### YOLOv3 Customization Example

```python
# configs/mmdet/yolov3_custom.py
_base_ = './yolov3.py'

# Change input size
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),  # 608→416
    dict(type='Pad', size=(416, 416), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# Change score threshold
model = dict(
    test_cfg=dict(
        score_thr=0.3,  # 0.05→0.3 (high confidence only)
    ),
)
```

### HRNetV2 Customization Example

```python
# configs/mmpose/hrnetv2_custom.py
_base_ = './hrnetv2.py'

# Change input size
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),  # 256→384 (higher resolution)
    heatmap_size=(96, 96),  # 64→96
    sigma=3,                # 2→3
)

# Update test pipeline
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]
```

### Using Custom Configuration

```python
from src.anime_face_detector import LandmarkDetector
from mmengine.config import Config

# Load custom configuration
custom_config = Config.fromfile('configs/mmdet/yolov3_custom.py')

detector = LandmarkDetector(
    landmark_detector_config_or_path='configs/mmpose/hrnetv2.py',
    landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
    face_detector_config_or_path=custom_config,
    face_detector_checkpoint_path='checkpoints/yolov3.pth',
)
```

---

## Model Selection Guide

### Face Detector Selection

| Requirement | Recommended Model | Reason |
|-------------|------------------|--------|
| Real-time processing | YOLOv3 | Fast |
| High accuracy needed | Faster R-CNN | High accuracy |
| Small face detection | YOLOv3 | Multi-scale detection |
| Memory constraints | YOLOv3 | Lower memory usage |

### Parameter Adjustment

| Parameter | Small Value | Large Value | Recommended |
|-----------|------------|-------------|-------------|
| box_scale_factor | Face region only | Wide context | 1.1-1.2 |
| flip_test | False (fast) | True (accurate) | True |
| score_thr | Many detections | High confidence only | 0.05-0.3 |

---

## Heatmap Visualization

HRNetV2 generates heatmaps for each landmark.

```python
import matplotlib.pyplot as plt
import numpy as np
from src.anime_face_detector import create_detector

detector = create_detector('yolov3')

# To access internal heatmaps, use the model directly
# (Normal API only returns coordinates)
```

Actual heatmaps have shape `(64, 64, 28)` where each channel corresponds to one landmark.

---

## Performance Comparison

### Face Detection

| Model | GPU (FPS) | CPU (FPS) | mAP | Size |
|-------|-----------|-----------|-----|------|
| YOLOv3 | 30-50 | 2-5 | 0.95 | 120MB |
| Faster R-CNN | 10-20 | 1-2 | 0.97 | 160MB |

### Landmark Detection

| Model | GPU (FPS) | CPU (FPS) | PCK@0.1 | Size |
|-------|-----------|-----------|---------|------|
| HRNetV2 | 50-100 | 5-10 | 0.95 | 100MB |

GPU: NVIDIA RTX 3090, CPU: Intel Core i9-10900K measurements (reference values).

---

## Troubleshooting

### Model Loading Error

```
RuntimeError: Error(s) in loading state_dict
```

Solution: The model file may be corrupted. Delete and re-download.

```bash
rm ~/.cache/torch/hub/checkpoints/mmdet_anime-face_*.pth
rm ~/.cache/torch/hub/checkpoints/mmpose_anime-face_*.pth
```

### Configuration File Error

```
KeyError: 'xxx is not in the config'
```

Solution: Check your mmdet/mmpose versions. This library supports mmdet 3.x and mmpose 1.x.

### CUDA Configuration Error

```
RuntimeError: CUDA error: no kernel image is available
```

Solution: Rebuild mmcv for your CUDA architecture.

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir
```

---

## Summary

anime-face-detector provides high-accuracy face detection with YOLOv3/Faster R-CNN and detailed 28-point landmark detection with HRNetV2. By customizing configuration files, you can achieve optimal performance for your use case.