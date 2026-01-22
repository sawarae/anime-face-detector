#!/usr/bin/env python3
"""
Training script for HRNetV2 anime face landmark detection.

Fine-tunes the existing HRNetV2 model on custom animal character data.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path to import anime_face_detector
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anime_face_detector.models.hrnetv2 import HRNetV2  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from dataset import AnimeFaceLandmarkDataset  # noqa: E402


class HeatmapLoss(nn.Module):
    """MSE loss for heatmap prediction."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Args:
            pred_heatmaps: Predicted heatmaps (B, 28, H, W)
            target_heatmaps: Target heatmaps (B, 28, H, W)
        """
        return self.criterion(pred_heatmaps, target_heatmaps)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        target_heatmaps = batch['heatmaps'].to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_heatmaps = model(images)

        # Compute loss
        loss = criterion(pred_heatmaps, target_heatmaps)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Log to tensorboard
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('train/loss', loss.item(), global_step)

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)

            # Forward pass
            pred_heatmaps = model(images)

            # Compute loss
            loss = criterion(pred_heatmaps, target_heatmaps)
            running_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(dataloader)
    writer.add_scalar('val/loss', avg_loss, epoch)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train HRNetV2 for anime face landmark detection')
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Directory containing training images'
    )
    parser.add_argument(
        '--annotation-dir',
        type=str,
        required=True,
        help='Directory containing annotations'
    )
    parser.add_argument(
        '--val-image-dir',
        type=str,
        help='Directory containing validation images (if different from train)'
    )
    parser.add_argument(
        '--val-annotation-dir',
        type=str,
        help='Directory containing validation annotations'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to pretrained checkpoint (for fine-tuning)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'runs')

    # Create datasets
    print('Creating datasets...')
    train_dataset = AnimeFaceLandmarkDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        input_size=(256, 256),
        heatmap_size=(64, 64),
        sigma=2.0,
    )

    if args.val_annotation_dir:
        val_dataset = AnimeFaceLandmarkDataset(
            image_dir=args.val_image_dir or args.image_dir,
            annotation_dir=args.val_annotation_dir,
            input_size=(256, 256),
            heatmap_size=(64, 64),
            sigma=2.0,
        )
    else:
        # Split training dataset for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

    # Create model
    print('Creating model...')
    model = HRNetV2(num_keypoints=28)

    # Load pretrained weights if provided
    if args.checkpoint:
        print(f'Loading pretrained weights from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        print('Loaded pretrained weights')

    model = model.to(args.device)

    # Setup optimizer and loss
    criterion = HeatmapLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, writer
        )
        print(f'Train Loss: {train_loss:.4f}')

        # Validate
        val_loss = validate(model, val_loader, criterion, args.device, epoch, writer)
        print(f'Val Loss: {val_loss:.4f}')

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f'Saved best model to {save_path}')

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f'Saved checkpoint to {save_path}')

    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    print(f'Saved final model to {final_path}')

    writer.close()
    print('Training complete!')


if __name__ == '__main__':
    main()
