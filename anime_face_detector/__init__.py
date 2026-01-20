import pathlib

import torch

from .detector import LandmarkDetector


def get_config_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'yolov8', 'hrnetv2']

    package_path = pathlib.Path(__file__).parent.resolve()
    if model_name in ['faster-rcnn', 'yolov3', 'yolov8']:
        config_dir = package_path / 'configs' / 'mmdet'
    else:
        config_dir = package_path / 'configs' / 'mmpose'
    return config_dir / f'{model_name}.py'


def get_checkpoint_path(model_name: str) -> pathlib.Path:
    """Get checkpoint path for pre-trained models.

    Note: YOLOv8 models should use custom_checkpoint_path parameter in create_detector()
    as there is no pre-trained YOLOv8 model provided by default.
    """
    assert model_name in ['faster-rcnn', 'yolov3', 'hrnetv2']
    if model_name in ['faster-rcnn', 'yolov3']:
        file_name = f'mmdet_anime-face_{model_name}.pth'
    else:
        file_name = f'mmpose_anime-face_{model_name}.pth'

    model_dir = pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / file_name
    if not model_path.exists():
        url = f'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/{file_name}'
        torch.hub.download_url_to_file(url, model_path.as_posix())

    return model_path


def create_detector(
    face_detector_name: str = 'yolov3',
    landmark_model_name: str = 'hrnetv2',
    device: str = 'cuda:0',
    flip_test: bool = True,
    box_scale_factor: float = 1.1,
    custom_detector_config_path: pathlib.Path | None = None,
    custom_detector_checkpoint_path: pathlib.Path | None = None,
    detector_framework: str | None = None,
) -> LandmarkDetector:
    """Create a landmark detector with face detection.

    Args:
        face_detector_name: Name of face detector ('yolov3', 'faster-rcnn', 'yolov8')
        landmark_model_name: Name of landmark model ('hrnetv2')
        device: Device to run models on ('cuda:0' or 'cpu')
        flip_test: Whether to use flip test for landmark detection
        box_scale_factor: Scale factor for detected face boxes
        custom_detector_config_path: Custom config path for face detector (required for mmdet yolov8)
        custom_detector_checkpoint_path: Custom checkpoint path for face detector
        detector_framework: Framework to use ('mmdet' or 'ultralytics').
                          If None, auto-detect based on checkpoint file extension
                          (.pth = mmdet, .pt = ultralytics)

    Returns:
        LandmarkDetector instance

    Examples:
        # Use pre-trained YOLOv3 (MMDetection)
        detector = create_detector('yolov3')

        # Use custom MMDetection YOLOv8 model
        detector = create_detector(
            'yolov8',
            custom_detector_config_path=pathlib.Path('path/to/yolov8_config.py'),
            custom_detector_checkpoint_path=pathlib.Path('path/to/yolov8_weights.pth')
        )

        # Use adetailer model (Ultralytics)
        detector = create_detector(
            custom_detector_checkpoint_path=pathlib.Path('face_yolov8n.pt'),
            detector_framework='ultralytics'
        )
    """
    assert face_detector_name in ['yolov3', 'faster-rcnn', 'yolov8', None]
    assert landmark_model_name in ['hrnetv2']

    # Auto-detect framework from checkpoint file extension if not specified
    if detector_framework is None and custom_detector_checkpoint_path is not None:
        checkpoint_str = str(custom_detector_checkpoint_path)
        if checkpoint_str.endswith('.pt'):
            detector_framework = 'ultralytics'
        elif checkpoint_str.endswith('.pth'):
            detector_framework = 'mmdet'
        else:
            detector_framework = 'mmdet'  # default
    elif detector_framework is None:
        detector_framework = 'mmdet'  # default for pre-trained models

    # Handle custom paths for face detector
    if detector_framework == 'ultralytics':
        # Ultralytics doesn't need config file
        detector_config_path = None
        if custom_detector_checkpoint_path is None:
            raise ValueError(
                "Ultralytics framework requires custom_detector_checkpoint_path "
                "to be specified (e.g., 'face_yolov8n.pt' from adetailer)"
            )
        detector_checkpoint_path = custom_detector_checkpoint_path
    else:
        # MMDetection framework
        if custom_detector_config_path is not None:
            detector_config_path = custom_detector_config_path
        else:
            if face_detector_name is None:
                face_detector_name = 'yolov3'  # default
            detector_config_path = get_config_path(face_detector_name)

        if custom_detector_checkpoint_path is not None:
            detector_checkpoint_path = custom_detector_checkpoint_path
        elif face_detector_name == 'yolov8':
            raise ValueError(
                "YOLOv8 requires a custom trained model. "
                "Please provide custom_detector_checkpoint_path parameter with path to your trained YOLOv8 model."
            )
        else:
            detector_checkpoint_path = get_checkpoint_path(face_detector_name)

    # Landmark model paths (always use pre-trained)
    landmark_config_path = get_config_path(landmark_model_name)
    landmark_checkpoint_path = get_checkpoint_path(landmark_model_name)

    model = LandmarkDetector(
        landmark_config_path,
        landmark_checkpoint_path,
        detector_config_path,
        detector_checkpoint_path,
        device=device,
        flip_test=flip_test,
        box_scale_factor=box_scale_factor,
        face_detector_framework=detector_framework,
    )
    return model
