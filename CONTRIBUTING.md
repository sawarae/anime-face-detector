# Contributing to Anime Face Detector

Thank you for your interest in contributing! This document provides guidelines for development using **uv**, the modern Python package manager.

## Development Setup

### Prerequisites

- Python 3.7 or later
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/hysts/anime-face-detector
cd anime-face-detector

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all development dependencies
make install-all
# Or manually:
uv pip install -e ".[all]"
```

## Using Make Commands

This project includes a Makefile for common development tasks:

```bash
# Show all available commands
make help

# Install dependencies
make install         # Runtime dependencies only
make install-dev     # + Development tools
make install-all     # Everything (runtime + dev + demo + conversion)

# Development
make format          # Format code with black and isort
make lint            # Run linters (flake8, mypy)
make test            # Run tests with pytest
make clean           # Clean build artifacts

# Demo and conversion
make demo            # Run Gradio demo
make convert-models  # Convert PyTorch models to ONNX
```

## Development Workflow

### 1. Code Formatting

We use **black** and **isort** for consistent code formatting:

```bash
# Format all code
make format

# Or manually
uv run black anime_face_detector/ tests/ tools/
uv run isort anime_face_detector/ tests/ tools/
```

### 2. Linting

Run linters before committing:

```bash
# Run all linters
make lint

# Or manually
uv run flake8 anime_face_detector/
uv run mypy anime_face_detector/
```

### 3. Testing

Add tests for new features:

```bash
# Run tests
make test

# Or manually
uv run pytest tests/ -v
```

### 4. Adding Dependencies

When adding new dependencies, update `pyproject.toml`:

```toml
[project]
dependencies = [
    "existing-package>=1.0.0",
    "new-package>=2.0.0",  # Add here
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "new-dev-tool>=1.0.0",  # Or add here for dev dependencies
]
```

Then reinstall:

```bash
uv pip install -e ".[all]"
```

## UV Commands Reference

### Essential Commands

```bash
# Create virtual environment
uv venv

# Install package in editable mode
uv pip install -e .

# Install with extras
uv pip install -e ".[dev]"
uv pip install -e ".[demo]"
uv pip install -e ".[all]"

# Add a new dependency
uv pip install package-name

# Upgrade dependencies
uv pip install --upgrade -e ".[all]"

# Sync dependencies (fast reinstall)
uv pip sync
```

### Advanced Usage

```bash
# Generate lock file for reproducibility
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip sync requirements.lock

# Run command in virtual environment
uv run pytest

# Create fresh environment
uv venv --seed
```

## Project Structure

```
anime-face-detector/
├── anime_face_detector/      # Main package
│   ├── __init__.py
│   ├── detector.py           # Core detection logic
│   ├── onnx_helper.py        # ONNX Runtime utilities
│   └── configs/              # Model configurations
├── tools/
│   └── convert_to_onnx.py    # Model conversion script
├── tests/                    # Test suite
├── pyproject.toml            # Project metadata and dependencies
├── Makefile                  # Development shortcuts
├── README.md                 # User documentation
└── CONTRIBUTING.md           # This file
```

## Commit Guidelines

1. **Format your code** before committing:
   ```bash
   make format
   ```

2. **Run linters** to catch issues:
   ```bash
   make lint
   ```

3. **Add tests** for new features

4. **Write clear commit messages**:
   ```
   Add ONNX support for faster inference

   - Implement ONNXFaceDetector and ONNXLandmarkDetector
   - Add conversion script for PyTorch to ONNX
   - Update documentation with ONNX usage
   ```

## Why UV?

We use [uv](https://github.com/astral-sh/uv) because it's:

- **⚡ 10-100x faster** than pip
- **🔒 More reliable** with deterministic dependency resolution
- **💾 More efficient** with smart caching
- **📦 Modern** with first-class pyproject.toml support
- **🎯 Compatible** as a drop-in replacement for pip

### Performance Comparison

| Task | pip | uv | Improvement |
|------|-----|-----|-------------|
| Install from cache | 2.5s | 0.05s | **50x faster** |
| Fresh install | 45s | 1.2s | **37x faster** |
| Dependency resolution | 15s | 0.3s | **50x faster** |

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/hysts/anime-face-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hysts/anime-face-detector/discussions)
- **UV Documentation**: [UV Docs](https://github.com/astral-sh/uv)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
