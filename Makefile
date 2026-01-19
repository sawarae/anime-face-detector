.PHONY: help install install-dev install-all test format lint clean demo convert-models

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies with uv
	uv pip install -e .

install-dev:  ## Install development dependencies with uv
	uv pip install -e ".[dev]"

install-all:  ## Install all dependencies (runtime + dev + demo + conversion)
	uv pip install -e ".[all]"

install-demo:  ## Install demo dependencies
	uv pip install -e ".[demo]"

install-conversion:  ## Install model conversion dependencies
	uv pip install -e ".[conversion]"

test:  ## Run tests with pytest
	uv run pytest tests/ -v

format:  ## Format code with black and isort
	uv run black anime_face_detector/ tests/ tools/
	uv run isort anime_face_detector/ tests/ tools/

lint:  ## Run linters (flake8, mypy)
	uv run flake8 anime_face_detector/ tests/ tools/
	uv run mypy anime_face_detector/

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo:  ## Run Gradio demo with ONNX
	python demo_gradio.py --use-onnx

demo-cpu:  ## Run Gradio demo on CPU
	python demo_gradio.py --use-onnx --device cpu

convert-models:  ## Convert PyTorch models to ONNX format
	@echo "Converting face detector (YOLOv3)..."
	python tools/convert_to_onnx.py --model yolov3
	@echo "Converting landmark detector (HRNetv2)..."
	python tools/convert_to_onnx.py --model hrnetv2
	@echo "Conversion complete!"

setup:  ## Initial setup: create venv and install dependencies
	uv venv
	@echo ""
	@echo "Virtual environment created!"
	@echo "Activate it with:"
	@echo "  source .venv/bin/activate  (Linux/macOS)"
	@echo "  .venv\\Scripts\\activate    (Windows)"
	@echo ""
	@echo "Then run 'make install' to install dependencies"

sync:  ## Sync dependencies with uv (faster reinstall)
	uv pip sync

upgrade:  ## Upgrade all dependencies to latest versions
	uv pip install --upgrade -e ".[all]"

build:  ## Build distribution packages
	uv build

publish:  ## Publish to PyPI (requires credentials)
	uv publish

lock:  ## Generate uv.lock file for reproducible builds
	uv pip compile pyproject.toml -o uv.lock

info:  ## Show project information
	@echo "Project: anime-face-detector"
	@echo "Python version: $$(python --version)"
	@echo "UV version: $$(uv --version)"
	@echo "Virtual environment: $${VIRTUAL_ENV:-Not activated}"
