#!/usr/bin/env python3
"""
Evaluation script for HRNetV2 anime face landmark detection.

Computes metrics like NME (Normalized Mean Error) on test set.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anime_face_detector.models.hrnetv2 import HRNetV2  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from dataset import AnimeFaceLandmarkDataset  # noqa: E402


def get_max_preds(heatmaps):
    """
    Get predictions from heatmaps using argmax.

    Args:
        heatmaps: (B, K, H, W) tensor

    Returns:
        preds: (B, K, 2) coordinates
        maxvals: (B, K, 1) confidence scores
    """
    batch_size, num_joints, height, width = heatmaps.shape

    # Reshape heatmaps
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_joints, -1)

    # Get max values and indices
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    # Convert to coordinates
    preds = torch.zeros(batch_size, num_joints, 2)
    preds[:, :, 0] = idx % width  # x
    preds[:, :, 1] = idx // width  # y

    maxvals = maxvals.unsqueeze(-1)

    return preds, maxvals


def compute_nme(pred_landmarks, gt_landmarks, normalization='bbox'):
    """
    Compute Normalized Mean Error (NME).

    Args:
        pred_landmarks: (B, K, 2) predicted landmarks
        gt_landmarks: (B, K, 2) ground truth landmarks
        normalization: 'bbox' or 'interocular'

    Returns:
        nme: Mean NME across all samples
    """
    batch_size = pred_landmarks.shape[0]
    errors = []

    for i in range(batch_size):
        pred = pred_landmarks[i]
        gt = gt_landmarks[i]

        # Compute Euclidean distance
        distances = np.sqrt(np.sum((pred - gt) ** 2, axis=1))

        # Normalization
        if normalization == 'bbox':
            # Normalize by bounding box size
            x_min, y_min = gt.min(axis=0)
            x_max, y_max = gt.max(axis=0)
            bbox_size = max(x_max - x_min, y_max - y_min)
            norm_distances = distances / bbox_size
        elif normalization == 'interocular':
            # Normalize by interocular distance (eyes)
            # Left eye: 11-13, Right eye: 17-19
            left_eye_center = gt[11:14].mean(axis=0)
            right_eye_center = gt[17:20].mean(axis=0)
            interocular = np.linalg.norm(left_eye_center - right_eye_center)
            norm_distances = distances / interocular
        else:
            raise ValueError(f'Unknown normalization: {normalization}')

        errors.append(norm_distances.mean())

    return np.mean(errors)


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()

    all_pred_landmarks = []
    all_gt_landmarks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            gt_landmarks = batch['landmarks'].cpu().numpy()

            # Forward pass
            pred_heatmaps = model(images)

            # Get predictions
            pred_coords, _ = get_max_preds(pred_heatmaps)

            # Scale predictions from heatmap size (64x64) to input size (256x256)
            pred_coords = pred_coords.cpu().numpy()
            pred_coords *= 4  # 256 / 64 = 4

            all_pred_landmarks.append(pred_coords)
            all_gt_landmarks.append(gt_landmarks)

    # Concatenate all predictions
    all_pred_landmarks = np.concatenate(all_pred_landmarks, axis=0)
    all_gt_landmarks = np.concatenate(all_gt_landmarks, axis=0)

    # Compute metrics
    nme_bbox = compute_nme(all_pred_landmarks, all_gt_landmarks, normalization='bbox')
    nme_interocular = compute_nme(all_pred_landmarks, all_gt_landmarks, normalization='interocular')

    return {
        'nme_bbox': nme_bbox,
        'nme_interocular': nme_interocular,
        'num_samples': len(all_pred_landmarks),
    }


def visualize_predictions(model, dataset, device, output_dir, num_samples=10):
    """Visualize predictions on sample images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0).to(device)
        gt_landmarks = sample['landmarks'].cpu().numpy()

        # Predict
        with torch.no_grad():
            pred_heatmaps = model(image)
            pred_coords, _ = get_max_preds(pred_heatmaps)
            pred_coords = pred_coords.cpu().numpy()[0] * 4  # Scale to 256x256

        # Convert image to numpy
        image_np = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Draw ground truth (green)
        for x, y in gt_landmarks:
            cv2.circle(image_np, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw predictions (red)
        for x, y in pred_coords:
            cv2.circle(image_np, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Save
        save_path = output_dir / f'sample_{i:03d}.jpg'
        cv2.imwrite(str(save_path), image_np)

    print(f'Saved {num_samples} visualizations to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate HRNetV2 model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--annotation-dir',
        type=str,
        required=True,
        help='Directory containing test annotations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eval_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualization of predictions'
    )
    parser.add_argument(
        '--num-vis-samples',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print('Creating dataset...')
    dataset = AnimeFaceLandmarkDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        input_size=(256, 256),
        heatmap_size=(64, 64),
        sigma=2.0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    print('Loading model...')
    model = HRNetV2(num_keypoints=28)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(args.device)

    # Evaluate
    print('Evaluating...')
    metrics = evaluate(model, dataloader, args.device)

    # Print results
    print('\n=== Evaluation Results ===')
    print(f'Number of samples: {metrics["num_samples"]}')
    print(f'NME (bbox): {metrics["nme_bbox"]:.4f}')
    print(f'NME (interocular): {metrics["nme_interocular"]:.4f}')

    # Save results
    results_path = output_dir / 'metrics.txt'
    with open(results_path, 'w') as f:
        f.write(f'Number of samples: {metrics["num_samples"]}\n')
        f.write(f'NME (bbox): {metrics["nme_bbox"]:.4f}\n')
        f.write(f'NME (interocular): {metrics["nme_interocular"]:.4f}\n')

    print(f'\nResults saved to {results_path}')

    # Visualize
    if args.visualize:
        print('\nGenerating visualizations...')
        visualize_predictions(
            model, dataset, args.device,
            output_dir / 'visualizations',
            num_samples=args.num_vis_samples
        )


if __name__ == '__main__':
    main()
