#!/usr/bin/env python3
"""Convert PyTorch models to ONNX format for faster inference."""

import argparse
import pathlib
import sys

import numpy as np
import torch

# Add parent directory to path to import anime_face_detector
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from anime_face_detector import get_checkpoint_path, get_config_path


def convert_face_detector_to_onnx(model_name: str, output_path: pathlib.Path,
                                  opset_version: int = 11):
    """Convert face detector (YOLO/Faster R-CNN) to ONNX.

    Args:
        model_name: Model name ('yolov3' or 'faster-rcnn')
        output_path: Output ONNX file path
        opset_version: ONNX opset version
    """
    try:
        from mmdet.apis import init_detector
    except ImportError:
        print("Error: mmdet is not installed. Please install it first.")
        return False

    print(f"Converting {model_name} face detector to ONNX...")

    # Load model
    config_path = get_config_path(model_name)
    checkpoint_path = get_checkpoint_path(model_name)

    model = init_detector(config_path, checkpoint_path, device='cpu')
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 800, 800)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path.as_posix(),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ Successfully exported to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False


def convert_landmark_detector_to_onnx(model_name: str, output_path: pathlib.Path,
                                     opset_version: int = 11):
    """Convert landmark detector (HRNet) to ONNX.

    Args:
        model_name: Model name ('hrnetv2')
        output_path: Output ONNX file path
        opset_version: ONNX opset version
    """
    try:
        from mmpose.apis import init_pose_model
    except ImportError:
        print("Error: mmpose is not installed. Please install it first.")
        return False

    print(f"Converting {model_name} landmark detector to ONNX...")

    # Load model
    config_path = get_config_path(model_name)
    checkpoint_path = get_checkpoint_path(model_name)

    model = init_pose_model(config_path, checkpoint_path, device='cpu')
    model.eval()

    # Create dummy input (face crop, typically 256x256)
    dummy_input = torch.randn(1, 3, 256, 256)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path.as_posix(),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ Successfully exported to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Export failed: {e}")
        print(f"Note: Direct ONNX export from mmpose models may not be fully supported.")
        print(f"You may need to use MMDeploy for proper conversion.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to ONNX format')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['yolov3', 'faster-rcnn', 'hrnetv2'],
        help='Model to convert')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: auto-generated)')
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version (default: 11)')

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_dir = pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
        output_dir.mkdir(exist_ok=True, parents=True)

        if args.model in ['yolov3', 'faster-rcnn']:
            output_path = output_dir / f'mmdet_anime-face_{args.model}.onnx'
        else:
            output_path = output_dir / f'mmpose_anime-face_{args.model}.onnx'
    else:
        output_path = pathlib.Path(args.output)

    # Convert model
    if args.model in ['yolov3', 'faster-rcnn']:
        success = convert_face_detector_to_onnx(args.model, output_path, args.opset)
    else:
        success = convert_landmark_detector_to_onnx(args.model, output_path, args.opset)

    if success:
        print(f"\n✓ Conversion completed successfully!")
        print(f"ONNX model saved to: {output_path}")
        print(f"Model size: {output_path.stat().st_size / (1024**2):.2f} MB")
    else:
        print(f"\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
