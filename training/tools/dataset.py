"""
PyTorch Dataset for anime face landmark detection training.
"""

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AnimeFaceLandmarkDataset(Dataset):
    """Dataset for anime face landmark detection."""

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        input_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 2.0,
        transform: Optional[Any] = None,
    ):
        """
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing JSON annotations
            input_size: Input image size (height, width)
            heatmap_size: Output heatmap size (height, width)
            sigma: Gaussian sigma for heatmap generation
            transform: Optional transforms to apply
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform

        # Load all annotations
        self.annotations = []
        for anno_path in sorted(self.annotation_dir.glob('*.json')):
            with open(anno_path) as f:
                data = json.load(f)
                # Verify image exists
                image_path = self.image_dir / data['image']
                if image_path.exists():
                    self.annotations.append(data)

        print(f'Loaded {len(self.annotations)} annotations from {annotation_dir}')

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - 'image': Tensor of shape (3, H, W)
            - 'heatmaps': Tensor of shape (28, H_hm, W_hm)
            - 'landmarks': Tensor of shape (28, 2)
        """
        anno = self.annotations[idx]

        # Load image
        image_path = self.image_dir / anno['image']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Failed to load image: {image_path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Get landmarks
        landmarks = np.array(anno['landmarks'], dtype=np.float32)  # (28, 2)

        # Resize image and scale landmarks
        image_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h
        landmarks_scaled = landmarks.copy()
        landmarks_scaled[:, 0] *= scale_x
        landmarks_scaled[:, 1] *= scale_y

        # Generate heatmaps
        heatmaps = self._generate_heatmaps(landmarks_scaled)

        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        landmarks_tensor = torch.from_numpy(landmarks_scaled).float()

        # Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return {
            'image': image_tensor,
            'heatmaps': heatmaps_tensor,
            'landmarks': landmarks_tensor,
        }

    def _generate_heatmaps(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmaps for landmarks.

        Args:
            landmarks: Array of shape (28, 2) containing (x, y) coordinates

        Returns:
            Heatmaps of shape (28, heatmap_h, heatmap_w)
        """
        num_landmarks = len(landmarks)
        heatmaps = np.zeros(
            (num_landmarks, self.heatmap_size[0], self.heatmap_size[1]),
            dtype=np.float32
        )

        # Scale landmarks to heatmap size
        scale_x = self.heatmap_size[1] / self.input_size[1]
        scale_y = self.heatmap_size[0] / self.input_size[0]

        for i, (x, y) in enumerate(landmarks):
            # Scale to heatmap coordinates
            hm_x = int(x * scale_x)
            hm_y = int(y * scale_y)

            # Skip if out of bounds
            if hm_x < 0 or hm_x >= self.heatmap_size[1] or hm_y < 0 or hm_y >= self.heatmap_size[0]:
                continue

            # Generate Gaussian heatmap
            heatmap = self._generate_gaussian(
                self.heatmap_size[0], self.heatmap_size[1],
                hm_x, hm_y, self.sigma
            )
            heatmaps[i] = heatmap

        return heatmaps

    def _generate_gaussian(
        self, height: int, width: int, center_x: int, center_y: int, sigma: float
    ) -> np.ndarray:
        """Generate a 2D Gaussian heatmap."""
        x = np.arange(0, width, 1, dtype=np.float32)
        y = np.arange(0, height, 1, dtype=np.float32)[:, np.newaxis]

        x0 = center_x
        y0 = center_y

        # Gaussian formula
        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return gaussian


def create_data_split(
    annotation_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split annotations into train/val/test sets.

    Args:
        annotation_dir: Directory containing JSON annotations
        output_dir: Directory to save split files
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed
    """
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all annotation files
    anno_files = sorted(annotation_dir.glob('*.json'))

    # Shuffle
    np.random.seed(seed)
    indices = np.random.permutation(len(anno_files))

    # Calculate split sizes
    n_total = len(anno_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Split
    train_files = [anno_files[i].name for i in indices[:n_train]]
    val_files = [anno_files[i].name for i in indices[n_train:n_train + n_val]]
    test_files = [anno_files[i].name for i in indices[n_train + n_val:]]

    # Save splits
    with open(output_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_files))

    with open(output_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_files))

    with open(output_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_files))

    print(f'Split complete: train={n_train}, val={n_val}, test={n_test}')
    print(f'Split files saved to {output_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create data split for training')
    parser.add_argument(
        '--annotation-dir',
        type=str,
        required=True,
        help='Directory containing JSON annotations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/datasets',
        help='Directory to save split files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training data ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation data ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test data ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    create_data_split(
        args.annotation_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
