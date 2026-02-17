# Development Guide

This document explains how to set up the development environment, coding conventions, testing methods, and contribution guidelines for anime-face-detector.

## Development Environment Setup

### Prerequisites

- Python 3.10 or 3.11
- CUDA 11.8 or later (when using GPU)
- Git
- uv (recommended) or pip

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/hysts/anime-face-detector.git
cd anime-face-detector
```

#### 2. Install Dependencies (using uv)

```bash
# System dependencies
sudo apt-get install -y ninja-build

# Create virtual environment
uv venv .venv && uv sync
uv pip install wheel

# Install xtcocoapi (mmpose dependency)
mkdir -p deps && cd deps
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi && ../../.venv/bin/python -m pip install -e . && cd ../..

# PyTorch (CUDA 12.8)
# For other CUDA versions: https://pytorch.org/get-started/previous-versions/
uv pip install torch==2.9.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

# OpenMMLab dependencies
uv pip install openmim mmengine

# mmcv (with CUDA operators)
# GPU architectures:
# - RTX 50XX (Blackwell): "12.0"
# - H100 (Hopper): "9.0"
# - RTX 40XX (Ada): "8.9"
# - RTX 30XX (Ampere): "8.0,8.6"
# - RTX 20XX (Turing): "7.5"
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir --no-build-isolation

# mmdet, mmpose
uv pip install --no-cache-dir mmdet==3.2.0 mmpose==1.3.2

# Development dependencies
uv pip install --no-cache-dir gradio ruff pre-commit
```

#### 3. Install in Editable Mode

```bash
pip install -e .
```

This allows source code changes to be immediately reflected.

#### 4. Set Up pre-commit Hooks

```bash
pre-commit install
```

---

## Project Structure

```
anime-face-detector/
├── anime_face_detector/          # Main package
│   ├── __init__.py               # Public API
│   ├── detector.py               # LandmarkDetector class
│   └── configs/                  # Model configurations
│       ├── mmdet/
│       │   ├── yolov3.py
│       │   └── faster-rcnn.py
│       └── mmpose/
│           └── hrnetv2.py
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── models_and_configs.md
│   └── development.md
├── assets/                       # Demo images
├── demo_gradio.py                # Gradio demo
├── demo.ipynb                    # Jupyter demo
├── pyproject.toml                # Project configuration
├── setup.py                      # Setup script
├── .pre-commit-config.yaml       # pre-commit configuration
├── CLAUDE.md                     # Claude Code guide
└── README.md                     # README file
```

---

## Coding Conventions

### Style Guide

This project uses **ruff** to maintain code quality.

#### ruff Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 88
indent-width = 4
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = []

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
line-ending = "lf"
```

### Coding Style

- **Strings**: Use single quotes (`'text'`)
- **Line length**: Maximum 88 characters
- **Indentation**: 4 spaces
- **Line endings**: LF (Unix style)
- **Import order**: Standard library → Third-party → Local

### Using ruff

```bash
# Lint check
ruff check .

# Auto-fix
ruff check . --fix

# Format
ruff format .

# Run all
ruff check . --fix && ruff format .
```

### pre-commit

ruff runs automatically before commits.

```bash
# Manual execution
pre-commit run --all-files

# Run specific hook only
pre-commit run ruff --all-files
```

---

## Type Hints

Please actively use Python 3.10+ type hints.

### Recommended Type Hints

```python
from __future__ import annotations

import pathlib
from typing import Union

import numpy as np
from mmengine.config import Config


def example_function(
    image_path: str | pathlib.Path,
    config: Config | None = None,
    threshold: float = 0.5,
) -> list[dict[str, np.ndarray]]:
    """Function description.

    Args:
        image_path: Path to image file
        config: Configuration object (optional)
        threshold: Threshold value (default: 0.5)

    Returns:
        List of detection results
    """
    pass
```

### Type Checking

Type checkers (mypy, etc.) are not currently used, but future adoption is under consideration.

---

## Testing

### Current State

Automated tests are not currently implemented. Please perform manual testing.

### Manual Testing

#### Basic Functionality Check

```python
from src.anime_face_detector import create_detector
import cv2

# Test with YOLOv3
detector = create_detector('yolov3', device='cpu')
image = cv2.imread('assets/input.jpg')
preds = detector(image)

assert len(preds) > 0, 'No faces detected'
assert preds[0]['keypoints'].shape == (28, 3), 'Invalid number of landmarks'
print('Test passed: YOLOv3')

# Test with Faster R-CNN
detector = create_detector('faster-rcnn', device='cpu')
preds = detector(image)

assert len(preds) > 0, 'No faces detected'
print('Test passed: Faster R-CNN')
```

#### Testing with Gradio Demo

```bash
python demo_gradio.py --device cpu
```

Open http://localhost:7860 in your browser to verify functionality through the UI.

### Future Test Implementation (TODO)

```bash
# pytest introduction planned
pip install pytest pytest-cov

# Run tests
pytest tests/

# Measure coverage
pytest --cov=anime_face_detector tests/
```

---

## Debugging

### Logging

You can use OpenMMLab's logging functionality.

```python
from mmengine.logging import print_log

print_log('Debug message', logger='current', level='DEBUG')
```

### Common Debug Points

#### 1. Check if models are loaded correctly

```python
detector = create_detector('yolov3')
print(f'Face detector: {detector.face_detector}')
print(f'Landmark detector: {detector.landmark_detector}')
```

#### 2. Check if configuration is applied correctly

```python
from mmengine.config import Config

config = Config.fromfile('anime_face_detector/src/configs/mmdet/yolov3.py')
print(config.pretty_text)
```

#### 3. Check if device is set correctly

```python
import torch

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
```

#### 4. Check inference scope

```python
from mmengine.registry import DefaultScope

print(f'Current scope: {DefaultScope.get_current_instance().scope_name}')
```

---

## Adding New Features

### Adding New Face Detectors

1. **Create configuration file**

```bash
touch anime_face_detector/src/configs/mmdet/new_detector.py
```

2. **Write configuration file**

```python
# anime_face_detector/src/configs/mmdet/new_detector.py
model = dict(
    type='YourDetectorType',
    # ... configuration
)

test_pipeline = [
    # ... pipeline
]

test_dataloader = dict(
    # ... dataloader
)
```

3. **Update __init__.py**

```python
# anime_face_detector/__init__.py
def get_config_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'new_detector', 'hrnetv2']  # Add
    # ...

def get_checkpoint_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'new_detector', 'hrnetv2']  # Add
    # ...

def create_detector(
    face_detector_name: str = 'yolov3',
    # ...
) -> LandmarkDetector:
    assert face_detector_name in ['yolov3', 'faster-rcnn', 'new_detector']  # Add
    # ...
```

4. **Place checkpoint**

```bash
# Upload to GitHub Releases or place locally
cp your_model.pth ~/.cache/torch/hub/checkpoints/mmdet_anime-face_new_detector.pth
```

### Adding New Landmark Models

Follow similar steps to add configuration files to `configs/mmpose/`.

---

## Documentation Updates

### Documentation Structure

- [architecture.md](architecture.md): Architecture explanation
- [api_reference.md](api_reference.md): API specifications
- [models_and_configs.md](../models_and_configs.md): Model and configuration details
- [development.md](development.md): This file

### Documentation Writing Rules

- Markdown format
- Include abundant code examples
- Written in English
- Use appropriate heading levels

---

## Release Process

### Versioning

Follows Semantic Versioning (SemVer).

```
MAJOR.MINOR.PATCH
Example: 1.2.3
```

- **MAJOR**: Incompatible changes
- **MINOR**: Backward-compatible feature additions
- **PATCH**: Backward-compatible bug fixes

### Release Steps (for Maintainers)

1. **Update version number**

```bash
# pyproject.toml
version = "0.1.0"

# setup.py
version='0.1.0'
```

2. **Update CHANGELOG.md**

```markdown
## [0.1.0] - 2026-01-20

### Added
- Description of new features

### Changed
- Changes made

### Fixed
- Bug fixes
```

3. **Create tag**

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

4. **Upload to PyPI**

```bash
python -m build
python -m twine upload dist/*
```

5. **Create GitHub Release**

- Write release notes
- Upload model files

---

## Building Docker Images

### Local Build

```bash
docker build -t anime-face-detector:latest .
```

### Multi-stage Build

The Dockerfile uses an optimized multi-stage build.

```dockerfile
# Build stage
FROM python:3.11-slim as builder
# ... install dependencies

# Runtime stage
FROM python:3.11-slim
# ... copy only necessary files
```

### Push to GitHub Container Registry

```bash
docker tag anime-face-detector:latest ghcr.io/username/anime-face-detector:latest
docker push ghcr.io/username/anime-face-detector:latest
```

---

## Contribution

### Reporting Issues

Report bugs and feature requests through GitHub Issues.

#### Bug Report Template

```markdown
## Bug Description
Briefly describe the bug.

## Steps to Reproduce
1. Execute xxx
2. Check yyy
3. Error occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Ubuntu 22.04
- Python: 3.11
- CUDA: 12.8
- anime-face-detector: 0.1.0
```

### Pull Requests

1. **Fork and Clone**

```bash
git clone https://github.com/your-username/anime-face-detector.git
cd anime-face-detector
git remote add upstream https://github.com/hysts/anime-face-detector.git
```

2. **Create Branch**

```bash
git checkout -b feature/my-new-feature
```

3. **Implement Changes**

- Follow coding conventions
- Pass pre-commit hooks
- Update documentation

4. **Commit**

```bash
git add .
git commit -m "Add my new feature"
```

Commit message format:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Refactoring
- `test`: Tests
- `chore`: Other

Example:
```
feat: Add support for custom landmark models

- Allow users to specify custom config and checkpoint
- Update documentation with usage examples

Closes #42
```

5. **Push**

```bash
git push origin feature/my-new-feature
```

6. **Create Pull Request**

Create a Pull Request on GitHub and describe your changes.

### Review Process

- Maintainers will review the code
- Request modifications if necessary
- Merge after approval

---

## Troubleshooting

### Common Development Environment Issues

#### mmcv Build Errors

```
error: cannot find -lcudart
```

Solution: Set CUDA_HOME.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### pre-commit Hook Failures

```
Ruff...........Failed
```

Solution: Auto-fix with ruff then recommit.

```bash
ruff check . --fix
ruff format .
git add .
git commit -m "Fix linting issues"
```

#### Gradio Demo Won't Start

```
ModuleNotFoundError: No module named 'gradio'
```

Solution: Install Gradio.

```bash
pip install gradio
```

---

## Performance Profiling

### Measuring Inference Speed

```python
import time
from src.anime_face_detector import create_detector
import cv2

detector = create_detector('yolov3', device='cuda:0')
image = cv2.imread('assets/input.jpg')

# Warmup
for _ in range(10):
    detector(image)

# Measurement
times = []
for _ in range(100):
    start = time.perf_counter()
    preds = detector(image)
    end = time.perf_counter()
    times.append(end - start)

import numpy as np

print(f'Average inference time: {np.mean(times) * 1000:.2f} ms')
print(f'Standard deviation: {np.std(times) * 1000:.2f} ms')
print(f'FPS: {1 / np.mean(times):.2f}')
```

### Measuring Memory Usage

```python
import torch
from src.anime_face_detector import create_detector

torch.cuda.reset_peak_memory_stats()

detector = create_detector('yolov3', device='cuda:0')
image = cv2.imread('assets/input.jpg')
preds = detector(image)

max_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
print(f'Maximum memory usage: {max_memory:.2f} MB')
```

---

## Additional Resources

### Official Documentation

- [mmdetection docs](https://mmdetection.readthedocs.io/)
- [mmpose docs](https://mmpose.readthedocs.io/)
- [mmengine docs](https://mmengine.readthedocs.io/)
- [mmcv docs](https://mmcv.readthedocs.io/)

### Community

- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, idea sharing

### Related Projects

- [OpenMMLab](https://github.com/open-mmlab)
- [anime-face-detector original](https://github.com/hysts/anime-face-detector)

---

## Summary

This document covers everything from setting up the development environment to contributing to anime-face-detector. If you have any questions, please ask through GitHub Issues or Discussions.

Happy coding!