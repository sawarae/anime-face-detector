from __future__ import annotations

import pathlib
from typing import Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXInferenceSession:
    """ONNX Runtime inference session wrapper for face and landmark detection."""

    def __init__(self, model_path: Union[str, pathlib.Path], device: str = 'cuda:0'):
        if not ONNX_AVAILABLE:
            raise ImportError(
                'onnxruntime is not installed. Please install it with: '
                'pip install onnxruntime-gpu (for GPU) or pip install onnxruntime (for CPU)')

        if isinstance(model_path, pathlib.Path):
            model_path = model_path.as_posix()

        # Set up execution providers based on device
        providers = self._get_providers(device)

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _get_providers(self, device: str) -> list[str]:
        """Get execution providers based on device."""
        if 'cuda' in device.lower():
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']

    def run(self, input_data: np.ndarray) -> list[np.ndarray]:
        """Run inference with ONNX Runtime.

        Args:
            input_data: Input tensor (numpy array)

        Returns:
            List of output tensors
        """
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        return outputs


class ONNXFaceDetector:
    """ONNX-based face detector."""

    def __init__(self, model_path: Union[str, pathlib.Path], device: str = 'cuda:0',
                 score_threshold: float = 0.3):
        self.session = ONNXInferenceSession(model_path, device)
        self.score_threshold = score_threshold

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Preprocess image for face detection.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image, scale factor, and original shape
        """
        orig_h, orig_w = image.shape[:2]

        # Resize image (typically to 800x600 for face detection)
        target_size = 800
        scale = target_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        resized_img = cv2.resize(image, (new_w, new_h))

        # Pad to standard size
        padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # Convert to RGB and normalize
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        normalized = rgb_img.astype(np.float32)
        normalized = (normalized - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])

        # CHW format
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        return input_tensor, scale, (orig_h, orig_w)

    def postprocess(self, outputs: list[np.ndarray], scale: float,
                   orig_shape: tuple[int, int]) -> np.ndarray:
        """Postprocess detection outputs.

        Args:
            outputs: Model outputs
            scale: Scale factor used in preprocessing
            orig_shape: Original image shape (h, w)

        Returns:
            Bounding boxes with format [x0, y0, x1, y1, score]
        """
        # Extract bboxes and scores from outputs
        # Format depends on model architecture (YOLO vs Faster R-CNN)
        boxes = outputs[0]  # Assuming first output contains boxes

        # Filter by score threshold
        if boxes.ndim == 3:
            boxes = boxes[0]  # Remove batch dimension

        # Scale boxes back to original image size
        if len(boxes) > 0 and boxes.shape[1] >= 5:
            boxes[:, [0, 2]] /= scale
            boxes[:, [1, 3]] /= scale

            # Filter by confidence
            mask = boxes[:, 4] >= self.score_threshold
            boxes = boxes[mask]

        return boxes

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Detect faces in image.

        Args:
            image: Input image in BGR format

        Returns:
            Bounding boxes with format [x0, y0, x1, y1, score]
        """
        input_tensor, scale, orig_shape = self.preprocess(image)
        outputs = self.session.run(input_tensor)
        boxes = self.postprocess(outputs, scale, orig_shape)
        return boxes


class ONNXLandmarkDetector:
    """ONNX-based landmark detector."""

    def __init__(self, model_path: Union[str, pathlib.Path], device: str = 'cuda:0'):
        self.session = ONNXInferenceSession(model_path, device)

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Preprocess image and bbox for landmark detection.

        Args:
            image: Input image in BGR format
            bbox: Bounding box [x0, y0, x1, y1, score]

        Returns:
            Preprocessed cropped image
        """
        x0, y0, x1, y1 = bbox[:4].astype(int)

        # Crop face region
        face_img = image[y0:y1, x0:x1]

        # Resize to model input size (typically 256x256 for HRNet)
        input_size = 256
        resized = cv2.resize(face_img, (input_size, input_size))

        # Convert to RGB and normalize
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_img.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # CHW format
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        return input_tensor

    def postprocess(self, heatmaps: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Postprocess heatmaps to get landmark coordinates.

        Args:
            heatmaps: Output heatmaps from model
            bbox: Original bounding box [x0, y0, x1, y1, score]

        Returns:
            Landmarks with format [x, y, score] for each keypoint
        """
        x0, y0, x1, y1 = bbox[:4]
        face_w, face_h = x1 - x0, y1 - y0

        # Get keypoint locations from heatmaps
        if heatmaps.ndim == 4:
            heatmaps = heatmaps[0]  # Remove batch dimension

        num_keypoints = heatmaps.shape[0]
        keypoints = np.zeros((num_keypoints, 3))

        for i in range(num_keypoints):
            heatmap = heatmaps[i]

            # Find max location
            max_idx = np.argmax(heatmap)
            h, w = heatmap.shape
            y, x = divmod(max_idx, w)

            # Get confidence score
            score = heatmap[y, x]

            # Scale to original image coordinates
            keypoints[i, 0] = x0 + (x / w) * face_w
            keypoints[i, 1] = y0 + (y / h) * face_h
            keypoints[i, 2] = float(score)

        return keypoints

    def __call__(self, image: np.ndarray, bboxes: list[np.ndarray]) -> list[dict]:
        """Detect landmarks for faces.

        Args:
            image: Input image in BGR format
            bboxes: List of bounding boxes

        Returns:
            List of detection results with bbox and keypoints
        """
        results = []

        for bbox in bboxes:
            input_tensor = self.preprocess(image, bbox)
            outputs = self.session.run(input_tensor)
            keypoints = self.postprocess(outputs[0], bbox)

            results.append({
                'bbox': bbox,
                'keypoints': keypoints
            })

        return results


def get_onnx_model_path(model_name: str) -> pathlib.Path:
    """Get ONNX model path, similar to get_checkpoint_path."""
    import os

    # Try to use torch.hub if available, otherwise use XDG cache
    try:
        import torch
        cache_dir = pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
    except ImportError:
        cache_home = os.environ.get('XDG_CACHE_HOME',
                                    os.path.join(os.path.expanduser('~'), '.cache'))
        cache_dir = pathlib.Path(cache_home) / 'anime_face_detector' / 'checkpoints'

    assert model_name in ['faster-rcnn', 'yolov3', 'hrnetv2']
    if model_name in ['faster-rcnn', 'yolov3']:
        file_name = f'mmdet_anime-face_{model_name}.onnx'
    else:
        file_name = f'mmpose_anime-face_{model_name}.onnx'

    cache_dir.mkdir(exist_ok=True, parents=True)
    model_path = cache_dir / file_name

    return model_path


def is_onnx_available() -> bool:
    """Check if ONNX Runtime is available."""
    return ONNX_AVAILABLE
