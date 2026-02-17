"""Quick test script for the new pure PyTorch implementation."""

import numpy as np

from anime_face_detector import create_detector


def test_new_implementation():
    # Create a dummy test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    print('Creating detector...')
    try:
        detector = create_detector(device='cpu')
        print('Detector created successfully!')

        print('\nRunning inference on test image...')
        results = detector(test_image)
        print(f'Inference successful! Detected {len(results)} faces.')

        if len(results) > 0:
            for i, result in enumerate(results):
                print(f'\nFace {i + 1}:')
                print(f'  Bbox: {result["bbox"]}')
                print(f'  Keypoints shape: {result["keypoints"].shape}')
                print(f'  First keypoint: {result["keypoints"][0]}')

    except Exception as e:
        print(f'Error: {e}')
        import traceback

        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_new_implementation ()
