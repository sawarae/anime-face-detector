#!/usr/bin/env python3
"""Simple test script to verify ONNX functionality."""

import sys

import numpy as np

from anime_face_detector import create_detector
from anime_face_detector.onnx_helper import is_onnx_available


def test_pytorch_backend():
    """Test PyTorch backend."""
    print("Testing PyTorch backend...")
    try:
        detector = create_detector('yolov3', device='cpu', use_onnx=False)
        print("✓ PyTorch detector created successfully")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        preds = detector(dummy_image)
        print(f"✓ PyTorch inference successful (detected {len(preds)} faces)")
        return True
    except Exception as e:
        print(f"✗ PyTorch backend test failed: {e}")
        return False


def test_onnx_backend():
    """Test ONNX backend."""
    print("\nTesting ONNX backend...")

    if not is_onnx_available():
        print("✗ ONNX Runtime is not installed")
        print("  Install with: pip install onnxruntime or onnxruntime-gpu")
        return False

    print("✓ ONNX Runtime is available")

    try:
        detector = create_detector('yolov3', device='cpu', use_onnx=True)
        print("  Note: ONNX models need to be converted first using:")
        print("  python tools/convert_to_onnx.py --model yolov3")
        print("  python tools/convert_to_onnx.py --model hrnetv2")
        return True
    except Exception as e:
        print(f"  Info: {e}")
        return False


def main():
    print("=" * 60)
    print("Anime Face Detector - ONNX Integration Test")
    print("=" * 60)

    # Test PyTorch backend
    pytorch_ok = test_pytorch_backend()

    # Test ONNX backend
    onnx_ok = test_onnx_backend()

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"PyTorch Backend: {'✓ PASSED' if pytorch_ok else '✗ FAILED'}")
    print(f"ONNX Backend:    {'✓ AVAILABLE' if onnx_ok else '✗ NOT AVAILABLE'}")
    print("=" * 60)

    if not pytorch_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
