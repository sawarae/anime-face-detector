# anime-face-detector Documentation

Technical documentation at the library level for anime-face-detector.

## Documentation Index

### 1. [Architecture](architecture.md)

Explains the internal architecture and implementation details of anime-face-detector.

**Main Topics:**
- System architecture and directory structure
- Two-stage detection pipeline (face detection → landmark detection)
- Integration with OpenMMLab 2.0
- Scope management and handling API changes
- Model initialization process
- Inference processing flow
- Performance optimization
- Error handling

**Recommended for:**
- Developers who want to understand the internal implementation
- Those considering customization or extension
- Those who want to know how to integrate with OpenMMLab

---

### 2. [API Reference](api_reference.md)

Detailed specifications for all public APIs.

**Main Topics:**
- `create_detector()`: Creating detectors
- `get_config_path()`: Getting configuration file paths
- `get_checkpoint_path()`: Getting checkpoint paths
- `LandmarkDetector`: Details of the main class
- Data formats and landmark indices
- Usage examples and troubleshooting

**Recommended for:**
- Developers using the library
- Those who want to check API specifications
- Those who want to know parameter details

---

### 3. [Models and Configuration Files](models_and_configs.md)

Detailed explanation of the models and configuration files used.

**Main Topics:**
- Model overview (YOLOv3, Faster R-CNN, HRNetV2)
- Detailed architecture of each model
- Configuration file structure and explanation
- 28-point landmark layout
- How to customize configuration files
- Model selection guide
- Performance comparison

**Recommended for:**
- Those who want to know model details
- Developers who want to customize configurations
- Those who want to select the optimal model

---

### 4. [Development Guide](development.md)

Guide from development environment setup to contribution.

**Main Topics:**
- Development environment setup
- Project structure
- Coding conventions (using ruff)
- Testing methods
- Debugging techniques
- How to add new features
- Documentation updates
- Release process
- Contribution guidelines

**Recommended for:**
- Developers who want to contribute to the project
- Those who want to set up a development environment
- Those who want to check coding conventions

---

## Quick Start

### Installation

```bash
pip install openmim
mim install mmengine mmcv mmdet mmpose
pip install anime-face-detector
```

### Basic Usage

```python
from src.anime_face_detector import create_detector
import cv2

# Create detector
detector = create_detector('yolov3', device='cuda:0')

# Detect faces and landmarks from image
image = cv2.imread('image.jpg')
preds = detector(image)

# Use results
for pred in preds:
    bbox = pred['bbox']  # [x0, y0, x1, y1, score]
    keypoints = pred['keypoints']  # [[x, y, score], ...] (28 points)
    print(f"Detected face: {bbox}")
    print(f"Number of landmarks: {len(keypoints)}")
```

See [API Reference](api_reference.md) for details.

---

## How to Read the Documentation

### For Beginners

1. First, learn the basic usage in [API Reference](api_reference.md)
2. Check the model selection method in [Models and Configuration Files](models_and_configs.md)
3. Understand the internal structure in [Architecture](architecture.md) as needed

### For Those Who Want to Customize

1. Get an overview in [Architecture](architecture.md)
2. Check how to customize configurations in [Models and Configuration Files](models_and_configs.md)
3. Set up the development environment in [Development Guide](development.md)

### For Contributors

1. Set up the development environment in [Development Guide](development.md)
2. Understand the internal implementation in [Architecture](architecture.md)
3. Develop according to coding conventions

---

## Documentation Structure

```
docs/
├── README.md               # This file (index)
├── architecture.md         # Architecture explanation
├── api_reference.md        # API reference
├── models_and_configs.md   # Models and configuration files
└── development.md          # Development guide
```

---

## Related Links

### Official Resources

- [GitHub Repository](https://github.com/hysts/anime-face-detector)
- [PyPI Package](https://pypi.org/project/anime-face-detector/)
- [Colab Demo](https://colab.research.google.com/github/hysts/anime-face-detector/blob/main/demo.ipynb)
- [Hugging Face Space](https://huggingface.co/spaces/ayousanz/anime-face-detector-gpu)

### OpenMMLab

- [mmdetection](https://github.com/open-mmlab/mmdetection) - Object detection framework
- [mmpose](https://github.com/open-mmlab/mmpose) - Pose estimation framework
- [mmengine](https://github.com/open-mmlab/mmengine) - Foundation framework
- [mmcv](https://github.com/open-mmlab/mmcv) - Computer vision library

### Community

- [Issues](https://github.com/hysts/anime-face-detector/issues) - Bug reports, feature requests
- [Discussions](https://github.com/hysts/anime-face-detector/discussions) - Questions, idea sharing

---

## FAQ

### Q: Which model should I use?

A: It depends on your use case.
- **Real-time processing**: YOLOv3 (fast)
- **High accuracy needed**: Faster R-CNN (high accuracy)

See [Models and Configuration Files - Model Selection Guide](models_and_configs.md#model-selection-guide) for details.

### Q: Does it work without GPU?

A: Yes. You can run on CPU by specifying `device='cpu'`.

```python
detector = create_detector('yolov3', device='cpu')
```

However, it will be more than 10 times slower compared to GPU.

### Q: Can I use custom models?

A: Yes. By using the `LandmarkDetector` class directly, you can specify custom configurations and checkpoints.

See [API Reference - Initialization with Custom Configuration](api_reference.md#initialization-with-custom-configuration) for details.

### Q: What is the landmark layout?

A: 28 landmarks are placed on each part of the face.

See [Models and Configuration Files - Landmark Layout Details](models_and_configs.md#landmark-layout-details) for details.

### Q: I got an error

A: For common errors, refer to:
- [API Reference - Errors and Troubleshooting](api_reference.md#errors-and-troubleshooting)
- [Models and Configuration Files - Troubleshooting](models_and_configs.md#troubleshooting)
- [Development Guide - Troubleshooting](development.md#troubleshooting)

---

## Contributing to Documentation

We accept pull requests for documentation improvement suggestions and typo corrections.

See [Development Guide - Contribution](development.md#contribution) for details.

---

## License

This documentation is published under the same MIT License as the anime-face-detector itself.

---

Last updated: 2026-01-20