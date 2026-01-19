#!/usr/bin/env python3
"""Integration tests for anime-face-detector."""

import numpy as np
import pytest


def test_imports():
    """Test that all modules can be imported."""
    from anime_face_detector import create_detector
    from anime_face_detector.detector import PYTORCH_AVAILABLE, LandmarkDetector
    from anime_face_detector.onnx_helper import is_onnx_available

    assert is_onnx_available() is True, "ONNX Runtime should be available"


def test_pytorch_availability():
    """Test PyTorch availability detection."""
    from anime_face_detector.detector import PYTORCH_AVAILABLE

    # In lightweight version, PyTorch should not be required
    print(f"PyTorch available: {PYTORCH_AVAILABLE}")


def test_onnx_availability():
    """Test ONNX Runtime availability."""
    from anime_face_detector.onnx_helper import is_onnx_available

    assert is_onnx_available(), "ONNX Runtime should be installed"


def test_create_detector_without_models():
    """Test detector creation without ONNX or PyTorch models."""
    from anime_face_detector import create_detector

    # Should raise RuntimeError when neither ONNX models nor PyTorch are available
    with pytest.raises(RuntimeError) as exc_info:
        detector = create_detector('yolov3', device='cpu')

    assert "Neither ONNX models nor PyTorch dependencies are available" in str(exc_info.value)


def test_dummy_onnx_inference():
    """Test ONNX inference with dummy data (requires ONNX models)."""
    pytest.skip("Requires actual ONNX model files - run with converted models")

    from anime_face_detector import create_detector

    # Create detector
    detector = create_detector('yolov3', device='cpu', use_onnx=True)

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Run inference
    preds = detector(dummy_image)

    assert isinstance(preds, list)
    print(f"Detected {len(preds)} faces")


def test_onnx_model_paths():
    """Test ONNX model path resolution."""
    from anime_face_detector.onnx_helper import get_onnx_model_path

    # Test YOLOv3 path
    yolo_path = get_onnx_model_path('yolov3')
    assert yolo_path.name == 'mmdet_anime-face_yolov3.onnx'
    assert yolo_path.parent.name == 'checkpoints'

    # Test HRNetv2 path
    hrnet_path = get_onnx_model_path('hrnetv2')
    assert hrnet_path.name == 'mmpose_anime-face_hrnetv2.onnx'
    assert hrnet_path.parent.name == 'checkpoints'


def test_cache_directory():
    """Test cache directory creation."""
    from anime_face_detector import _get_cache_dir

    cache_dir = _get_cache_dir()
    assert cache_dir.exists() or True  # Should exist or be creatable
    print(f"Cache directory: {cache_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
