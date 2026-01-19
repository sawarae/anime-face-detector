import pathlib
import os

# Try to import torch (optional for ONNX-only mode)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .detector import LandmarkDetector


def get_config_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'hrnetv2']

    package_path = pathlib.Path(__file__).parent.resolve()
    if model_name in ['faster-rcnn', 'yolov3']:
        config_dir = package_path / 'configs' / 'mmdet'
    else:
        config_dir = package_path / 'configs' / 'mmpose'
    return config_dir / f'{model_name}.py'


def _get_cache_dir() -> pathlib.Path:
    """Get cache directory for models."""
    if TORCH_AVAILABLE:
        return pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
    else:
        # Use XDG cache dir or fallback to ~/.cache
        cache_home = os.environ.get('XDG_CACHE_HOME',
                                    os.path.join(os.path.expanduser('~'), '.cache'))
        return pathlib.Path(cache_home) / 'anime_face_detector' / 'checkpoints'


def _download_file(url: str, destination: pathlib.Path):
    """Download file from URL."""
    if TORCH_AVAILABLE:
        torch.hub.download_url_to_file(url, destination.as_posix())
    else:
        import urllib.request
        print(f"Downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination.as_posix())


def get_checkpoint_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'hrnetv2']
    if model_name in ['faster-rcnn', 'yolov3']:
        file_name = f'mmdet_anime-face_{model_name}.pth'
    else:
        file_name = f'mmpose_anime-face_{model_name}.pth'

    model_dir = _get_cache_dir()
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / file_name
    if not model_path.exists():
        url = f'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/{file_name}'
        _download_file(url, model_path)

    return model_path


def create_detector(face_detector_name: str = 'yolov3',
                    landmark_model_name='hrnetv2',
                    device: str = 'cuda:0',
                    flip_test: bool = True,
                    box_scale_factor: float = 1.1,
                    use_onnx: bool = None) -> LandmarkDetector:
    """Create a landmark detector with optional ONNX acceleration.

    Args:
        face_detector_name: Face detector model ('yolov3' or 'faster-rcnn')
        landmark_model_name: Landmark model ('hrnetv2')
        device: Device to run on ('cuda:0' or 'cpu')
        flip_test: Whether to use flip test for landmark detection
        box_scale_factor: Scale factor for bounding boxes
        use_onnx: Whether to use ONNX models for faster inference
                  (default: True if PyTorch not available, False otherwise)

    Returns:
        LandmarkDetector instance
    """
    assert face_detector_name in ['yolov3', 'faster-rcnn']
    assert landmark_model_name in ['hrnetv2']

    # Auto-detect ONNX mode if not specified
    if use_onnx is None:
        use_onnx = not TORCH_AVAILABLE

    detector_config_path = get_config_path(face_detector_name) if TORCH_AVAILABLE else None
    landmark_config_path = get_config_path(landmark_model_name) if TORCH_AVAILABLE else None
    detector_checkpoint_path = get_checkpoint_path(face_detector_name) if not use_onnx else None
    landmark_checkpoint_path = get_checkpoint_path(landmark_model_name) if not use_onnx else None

    model = LandmarkDetector(landmark_config_path,
                             landmark_checkpoint_path,
                             detector_config_path,
                             detector_checkpoint_path,
                             device=device,
                             flip_test=flip_test,
                             box_scale_factor=box_scale_factor,
                             use_onnx=use_onnx,
                             face_detector_name=face_detector_name,
                             landmark_detector_name=landmark_model_name)
    return model
