# API Reference

This document provides detailed explanations of all public APIs for anime-face-detector.

## Module: anime_face_detector

### create_detector()

Creates a detector instance.

```python
def create_detector(
    face_detector_name: str = 'yolov3',
    landmark_model_name: str = 'hrnetv2',
    device: str = 'cuda:0',
    flip_test: bool = True,
    box_scale_factor: float = 1.1,
) -> LandmarkDetector
```

#### Parameters

- **face_detector_name** (`str`, default: `'yolov3'`)
  - Model used for face detection
  - Options: `'yolov3'`, `'faster-rcnn'`
  - `'yolov3'`: Fast, suitable for real-time applications
  - `'faster-rcnn'`: High accuracy, recommended for precision-focused use cases

- **landmark_model_name** (`str`, default: `'hrnetv2'`)
  - Model used for landmark detection
  - Currently only `'hrnetv2'` is supported

- **device** (`str`, default: `'cuda:0'`)
  - Device to use for inference
  - Examples: `'cuda:0'`, `'cuda:1'`, `'cpu'`
  - `'cuda:0'` is recommended if CUDA-enabled GPU is available

- **flip_test** (`bool`, default: `True`)
  - Whether to use horizontal flip test during landmark detection
  - `True`: Uses flipped image results to improve accuracy (inference time approximately doubles)
  - `False`: Only normal inference (faster)

- **box_scale_factor** (`float`, default: `1.1`)
  - Bounding box scaling factor after face detection
  - Larger values use wider area around the face
  - Recommended range: 1.0 - 1.3
  - 1.0: No scaling, 1.1: 10% scaling, 1.2: 20% scaling

#### Returns

- `LandmarkDetector`: Instance that performs face detection and landmark detection

#### Exceptions

- `AssertionError`: When an invalid model name is specified

#### Usage Examples

```python
from src.anime_face_detector import create_detector

# Default settings (YOLOv3, CUDA)
detector = create_detector()

# Using Faster R-CNN, running on CPU
detector = create_detector(
  face_detector_name='faster-rcnn',
  device='cpu'
)

# Disable flip_test for speed
detector = create_detector(
  flip_test=False,
  box_scale_factor=1.2
)
```

---

### get_config_path()

Retrieves the configuration file path for a model.

```python
def get_config_path(model_name: str) -> pathlib.Path
```

#### Parameters

- **model_name** (`str`)
  - Model name
  - Options: `'yolov3'`, `'faster-rcnn'`, `'hrnetv2'`

#### Returns

- `pathlib.Path`: Absolute path to the configuration file

#### Exceptions

- `AssertionError`: When an invalid model name is specified

#### Usage Examples

```python
from src.anime_face_detector import get_config_path

config_path = get_config_path('yolov3')
print(config_path)
# /path/to/anime_face_detector/src/configs/mmdet/yolov3.py
```

---

### get_checkpoint_path()

Retrieves the checkpoint file path for a model. If the file doesn't exist, it's automatically downloaded.

```python
def get_checkpoint_path(model_name: str) -> pathlib.Path
```

#### Parameters

- **model_name** (`str`)
  - Model name
  - Options: `'yolov3'`, `'faster-rcnn'`, `'hrnetv2'`

#### Returns

- `pathlib.Path`: Absolute path to the checkpoint file

#### Exceptions

- `AssertionError`: When an invalid model name is specified

#### Behavior

1. Checks if file exists in `torch.hub.get_dir() + '/checkpoints/'`
2. If not, downloads from GitHub Releases
3. Returns the path

Default download location: `~/.cache/torch/hub/checkpoints/`

#### Usage Examples

```python
from src.anime_face_detector import get_checkpoint_path

checkpoint_path = get_checkpoint_path('yolov3')
print(checkpoint_path)
# /home/user/.cache/torch/hub/checkpoints/mmdet_anime-face_yolov3.pth
```

---

## Class: LandmarkDetector

A class that integrates face detection and landmark detection.

### Constructor

```python
def __init__(
    self,
    landmark_detector_config_or_path: Config | str | pathlib.Path,
    landmark_detector_checkpoint_path: str | pathlib.Path,
    face_detector_config_or_path: Config | str | pathlib.Path | None = None,
    face_detector_checkpoint_path: str | pathlib.Path | None = None,
    device: str = 'cuda:0',
    flip_test: bool = True,
    box_scale_factor: float = 1.1,
)
```

#### Parameters

- **landmark_detector_config_or_path** (`Config | str | pathlib.Path`)
  - Configuration for landmark detector (Config object or file path)

- **landmark_detector_checkpoint_path** (`str | pathlib.Path`)
  - Checkpoint file path for landmark detector

- **face_detector_config_or_path** (`Config | str | pathlib.Path | None`, default: `None`)
  - Configuration for face detector (Config object or file path)
  - If `None`, face detection is not used

- **face_detector_checkpoint_path** (`str | pathlib.Path | None`, default: `None`)
  - Checkpoint file path for face detector

- **device** (`str`, default: `'cuda:0'`)
  - Device to use for inference

- **flip_test** (`bool`, default: `True`)
  - Whether to use horizontal flip test during landmark detection

- **box_scale_factor** (`float`, default: `1.1`)
  - Bounding box scaling factor

#### Usage Examples

```python
from src.anime_face_detector import LandmarkDetector
from mmengine.config import Config

# Specify configuration file paths
detector = LandmarkDetector(
  landmark_detector_config_or_path='configs/mmpose/hrnetv2.py',
  landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
  face_detector_config_or_path='configs/mmdet/yolov3.py',
  face_detector_checkpoint_path='checkpoints/yolov3.pth',
  device='cuda:0'
)

# Use Config objects directly
landmark_config = Config.fromfile('configs/mmpose/hrnetv2.py')
face_config = Config.fromfile('configs/mmdet/yolov3.py')

detector = LandmarkDetector(
  landmark_detector_config_or_path=landmark_config,
  landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
  face_detector_config_or_path=face_config,
  face_detector_checkpoint_path='checkpoints/yolov3.pth'
)
```

---

### \_\_call\_\_()

Detects faces from an image and estimates landmarks.

```python
def __call__(
    self,
    image_or_path: np.ndarray | str | pathlib.Path,
    boxes: list[np.ndarray] | None = None,
) -> list[dict[str, np.ndarray]]
```

#### Parameters

- **image_or_path** (`np.ndarray | str | pathlib.Path`)
  - Input image
  - `np.ndarray`: Image array in BGR format (OpenCV format)
  - `str` or `pathlib.Path`: Path to image file

- **boxes** (`list[np.ndarray] | None`, default: `None`)
  - List of face bounding boxes
  - Each box is in format `[x0, y0, x1, y1]` or `[x0, y0, x1, y1, score]`
  - If `None`, automatically detects using face detector
  - If face detector is not specified, uses entire image as face region

#### Returns

- `list[dict[str, np.ndarray]]`: List of detection results

Each element is a dictionary with the following structure:

```python
{
    'bbox': np.ndarray,      # shape: (5,), [x0, y0, x1, y1, score]
    'keypoints': np.ndarray  # shape: (28, 3), [[x, y, score], ...]
}
```

- `bbox`: Face bounding box
  - `x0, y0`: Top-left coordinates
  - `x1, y1`: Bottom-right coordinates
  - `score`: Detection confidence (0.0 - 1.0)

- `keypoints`: 28 landmark coordinates
  - Each landmark is in format `[x, y, score]`
  - `x, y`: Pixel coordinates
  - `score`: Keypoint confidence (0.0 - 1.0)

#### Landmark Indices

```
0-2:   Left side face contour
3-5:   Right side face contour
5-7:   Left eyebrow
8-10:  Right eyebrow
11-13: Left eye
14-16: Nose
17-19: Right eye
20-22: Mouth (right side)
23:    Mouth center bottom
24-26: Lower mouth
27:    Mouth center
```

#### Usage Examples

```python
import cv2
from src.anime_face_detector import create_detector

detector = create_detector('yolov3')

# Detect from image file path
preds = detector('path/to/image.jpg')

# Detect from NumPy array
image = cv2.imread('path/to/image.jpg')
preds = detector(image)

# Use pre-detected face boxes
boxes = [np.array([100, 100, 300, 300])]
preds = detector(image, boxes=boxes)

# Using results
for pred in preds:
  bbox = pred['bbox']
  keypoints = pred['keypoints']

  print(f"Face position: ({bbox[0]:.1f}, {bbox[1]:.1f}) - ({bbox[2]:.1f}, {bbox[3]:.1f})")
  print(f"Detection confidence: {bbox[4]:.3f}")

  for i, (x, y, score) in enumerate(keypoints):
    print(f"Landmark {i}: ({x:.1f}, {y:.1f}), confidence: {score:.3f}")
```

#### Drawing on Image Example

```python
import cv2
import numpy as np
from src.anime_face_detector import create_detector

detector = create_detector('yolov3')
image = cv2.imread('input.jpg')
preds = detector(image)

# Draw bounding boxes
for pred in preds:
  bbox = pred['bbox'].astype(int)
  cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

  # Draw landmarks
  keypoints = pred['keypoints']
  for x, y, score in keypoints:
    if score > 0.5:  # Only draw landmarks with confidence > 0.5
      cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

cv2.imwrite('output.jpg', image)
```

---

## Data Formats

### Bounding Box Format

```python
bbox: np.ndarray  # shape: (5,), dtype: float32
# [x0, y0, x1, y1, score]
```

- `x0, y0`: Top-left pixel coordinates
- `x1, y1`: Bottom-right pixel coordinates
- `score`: Detection confidence (0.0 - 1.0)

### Keypoint Format

```python
keypoints: np.ndarray  # shape: (28, 3), dtype: float32
# [[x0, y0, score0],
#  [x1, y1, score1],
#  ...
#  [x27, y27, score27]]
```

- `x, y`: Keypoint pixel coordinates
- `score`: Keypoint confidence (0.0 - 1.0)

---

## Internal Methods (For Reference)

The following methods are internal implementations and typically don't need to be called directly.

### _detect_faces()

```python
def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]
```

Detects faces using mmdet.

### _detect_landmarks()

```python
def _detect_landmarks(
    self, image: np.ndarray, boxes: list[np.ndarray]
) -> list[dict[str, np.ndarray]]
```

Detects landmarks using mmpose.

### _load_image()

```python
@staticmethod
def _load_image(image_or_path: np.ndarray | str | pathlib.Path) -> np.ndarray
```

Loads an image and converts it to a NumPy array.

### _update_pred_box()

```python
def _update_pred_box(self, pred_boxes: np.ndarray) -> list[np.ndarray]
```

Scales bounding boxes by `box_scale_factor`.

---

## Advanced Usage Examples

### Initialization with Custom Configuration

```python
from src.anime_face_detector import LandmarkDetector
from mmengine.config import Config

# Create custom configuration
landmark_config = Config.fromfile('configs/mmpose/hrnetv2.py')
landmark_config.model.test_cfg.flip_test = False  # Disable flip_test

detector = LandmarkDetector(
  landmark_detector_config_or_path=landmark_config,
  landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
  device='cuda:0'
)
```

### Batch Processing

```python
import glob
from src.anime_face_detector import create_detector

detector = create_detector('yolov3')

image_paths = glob.glob('images/*.jpg')
for image_path in image_paths:
  preds = detector(image_path)
  # Process...
```

### Filtering by Confidence

```python
from src.anime_face_detector import create_detector

detector = create_detector('yolov3')
preds = detector('image.jpg')

# Use only faces with confidence > 0.9
high_conf_preds = [pred for pred in preds if pred['bbox'][4] > 0.9]

# Use only landmarks with confidence > 0.5
for pred in preds:
  keypoints = pred['keypoints']
  reliable_keypoints = keypoints[keypoints[:, 2] > 0.5]
```

---

## Errors and Troubleshooting

### Common Errors

#### AssertionError: Invalid model name

```python
detector = create_detector('invalid-model')
# AssertionError: assertion failed
```

Solution: Use supported model names (`'yolov3'`, `'faster-rcnn'`, `'hrnetv2'`).

#### CUDA out of memory

```python
# RuntimeError: CUDA out of memory
```

Solutions:
- Use smaller image sizes
- Switch to CPU mode with `device='cpu'`
- Terminate other CUDA processes

#### KeyError: 'xxx is not in the xxx registry'

Registry scope error. Usually handled automatically internally, but may occur during custom implementations.

Solution: Use `DefaultScope.overwrite_default_scope()` appropriately.

---

## Type Hints

```python
from typing import Union
import pathlib
import numpy as np
from src.anime_face_detector import LandmarkDetector

ImageInput = Union[np.ndarray, str, pathlib.Path]
BoundingBox = np.ndarray  # shape: (5,)
Keypoints = np.ndarray  # shape: (28, 3)
Detection = dict[str, np.ndarray]  # {'bbox': BoundingBox, 'keypoints': Keypoints}
```
