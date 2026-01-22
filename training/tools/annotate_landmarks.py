#!/usr/bin/env python3
"""
Gradio-based annotation tool for marking 28 facial landmarks on anime characters.

Landmark points (0-indexed):
- 0-4: Face contour
- 5-10: Eyebrows
- 11-13, 17-19: Eyes
- 14-16: Nose
- 20-27: Mouth
"""

import json
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Define landmark names for reference
LANDMARK_NAMES = [
    # Face contour (0-4)
    'face_0', 'face_1', 'face_2', 'face_3', 'face_4',
    # Eyebrows (5-10)
    'left_eyebrow_0', 'left_eyebrow_1', 'left_eyebrow_2',
    'right_eyebrow_0', 'right_eyebrow_1', 'right_eyebrow_2',
    # Left eye (11-13)
    'left_eye_0', 'left_eye_1', 'left_eye_2',
    # Nose (14-16)
    'nose_0', 'nose_1', 'nose_2',
    # Right eye (17-19)
    'right_eye_0', 'right_eye_1', 'right_eye_2',
    # Mouth (20-27)
    'mouth_0', 'mouth_1', 'mouth_2', 'mouth_3',
    'mouth_4', 'mouth_5', 'mouth_6', 'mouth_7',
]

# Colors for different landmark groups (BGR format)
LANDMARK_COLORS = {
    'face': (255, 0, 0),      # Blue
    'eyebrow': (0, 255, 0),   # Green
    'eye': (0, 255, 255),     # Yellow
    'nose': (255, 0, 255),    # Magenta
    'mouth': (0, 0, 255),     # Red
}


def get_landmark_color(idx: int) -> tuple:
    """Get color for a landmark based on its index."""
    if 0 <= idx <= 4:
        return LANDMARK_COLORS['face']
    elif 5 <= idx <= 10:
        return LANDMARK_COLORS['eyebrow']
    elif 11 <= idx <= 13 or 17 <= idx <= 19:
        return LANDMARK_COLORS['eye']
    elif 14 <= idx <= 16:
        return LANDMARK_COLORS['nose']
    elif 20 <= idx <= 27:
        return LANDMARK_COLORS['mouth']
    return (128, 128, 128)  # Gray for unknown


class AnnotationTool:
    def __init__(self, image_dir: Path, output_dir: Path):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get list of images
        self.image_files = sorted(
            list(self.image_dir.glob('*.jpg'))
            + list(self.image_dir.glob('*.png'))
            + list(self.image_dir.glob('*.jpeg'))
        )
        self.current_idx = 0

        # Current annotations
        self.landmarks = []  # List of [x, y] coordinates
        self.current_image = None
        self.current_image_name = None

    def load_image(self, idx: int) -> tuple[np.ndarray, str]:
        """Load an image by index."""
        if not self.image_files or idx >= len(self.image_files):
            return None, 'No more images'

        self.current_idx = idx
        image_path = self.image_files[idx]
        self.current_image_name = image_path.name

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, f'Failed to load {image_path.name}'

        self.current_image = image.copy()

        # Load existing annotations if available
        anno_path = self.output_dir / f'{image_path.stem}.json'
        if anno_path.exists():
            with open(anno_path) as f:
                data = json.load(f)
                self.landmarks = data.get('landmarks', [])
        else:
            self.landmarks = []

        return self.render_image(), f'Image {idx + 1}/{len(self.image_files)}: {self.current_image_name}'

    def render_image(self) -> np.ndarray:
        """Render current image with landmarks."""
        if self.current_image is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)

        image = self.current_image.copy()

        # Draw existing landmarks
        for i, (x, y) in enumerate(self.landmarks):
            color = get_landmark_color(i)
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            cv2.putText(
                image, str(i), (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Draw connections (optional, for visualization)
        if len(self.landmarks) >= 28:
            # Face contour
            for i in range(4):
                pt1 = tuple(map(int, self.landmarks[i]))
                pt2 = tuple(map(int, self.landmarks[i + 1]))
                cv2.line(image, pt1, pt2, LANDMARK_COLORS['face'], 1)

            # Left eyebrow
            for i in range(5, 7):
                pt1 = tuple(map(int, self.landmarks[i]))
                pt2 = tuple(map(int, self.landmarks[i + 1]))
                cv2.line(image, pt1, pt2, LANDMARK_COLORS['eyebrow'], 1)

            # Right eyebrow
            for i in range(8, 10):
                pt1 = tuple(map(int, self.landmarks[i]))
                pt2 = tuple(map(int, self.landmarks[i + 1]))
                cv2.line(image, pt1, pt2, LANDMARK_COLORS['eyebrow'], 1)

            # Mouth
            mouth_pts = np.array([self.landmarks[20:28]], dtype=np.int32)
            cv2.polylines(image, mouth_pts, True, LANDMARK_COLORS['mouth'], 1)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def add_landmark(self, evt: gr.SelectData) -> tuple[np.ndarray, str]:
        """Add a landmark at the clicked position."""
        if self.current_image is None:
            return self.render_image(), 'No image loaded'

        x, y = evt.index

        if len(self.landmarks) < 28:
            self.landmarks.append([x, y])
            status = f'Added landmark {len(self.landmarks) - 1} ({LANDMARK_NAMES[len(self.landmarks) - 1]}): ({x}, {y})'
        else:
            status = 'All 28 landmarks already placed. Use "Clear Last" or "Clear All" to modify.'

        return self.render_image(), status

    def clear_last(self) -> tuple[np.ndarray, str]:
        """Remove the last landmark."""
        if self.landmarks:
            removed = self.landmarks.pop()
            return self.render_image(), f'Removed landmark {len(self.landmarks)}: {removed}'
        return self.render_image(), 'No landmarks to remove'

    def clear_all(self) -> tuple[np.ndarray, str]:
        """Clear all landmarks."""
        self.landmarks = []
        return self.render_image(), 'Cleared all landmarks'

    def save_annotation(self) -> str:
        """Save current annotations to JSON."""
        if self.current_image_name is None:
            return 'No image loaded'

        if len(self.landmarks) != 28:
            return f'Error: Need exactly 28 landmarks, but got {len(self.landmarks)}'

        # Save annotation
        anno_path = self.output_dir / f'{Path(self.current_image_name).stem}.json'
        data = {
            'image': self.current_image_name,
            'landmarks': self.landmarks,
            'image_size': {
                'width': self.current_image.shape[1],
                'height': self.current_image.shape[0]
            }
        }

        with open(anno_path, 'w') as f:
            json.dump(data, f, indent=2)

        return f'Saved annotation to {anno_path}'

    def next_image(self) -> tuple[np.ndarray, str]:
        """Load next image."""
        return self.load_image(self.current_idx + 1)

    def prev_image(self) -> tuple[np.ndarray, str]:
        """Load previous image."""
        return self.load_image(max(0, self.current_idx - 1))


def create_interface(image_dir: str, output_dir: str):
    """Create Gradio interface for annotation."""
    tool = AnnotationTool(Path(image_dir), Path(output_dir))

    with gr.Blocks(title='Anime Face Landmark Annotation Tool') as demo:
        gr.Markdown('# Anime Face Landmark Annotation Tool')
        gr.Markdown(
            """
Click on the image to add landmarks in order (0-27).

**Landmark Order:**
- 0-4: Face contour (blue)
- 5-10: Eyebrows (green)
- 11-13, 17-19: Eyes (yellow)
- 14-16: Nose (magenta)
- 20-27: Mouth (red)
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                image_display = gr.Image(
                    label='Click to add landmarks',
                    type='numpy',
                    interactive=True
                )
            with gr.Column(scale=1):
                status_text = gr.Textbox(label='Status', interactive=False)

                with gr.Row():
                    prev_btn = gr.Button('⬅️ Previous')
                    next_btn = gr.Button('Next ➡️')

                with gr.Row():
                    clear_last_btn = gr.Button('Clear Last', variant='secondary')
                    clear_all_btn = gr.Button('Clear All', variant='secondary')

                save_btn = gr.Button('💾 Save Annotation', variant='primary')

                gr.Markdown(
                    f"""
### Info
- Total images: {len(tool.image_files)}
- Image dir: `{image_dir}`
- Output dir: `{output_dir}`
"""
                )

        # Event handlers
        demo.load(tool.load_image, inputs=[gr.Number(value=0, visible=False)], outputs=[image_display, status_text])
        image_display.select(tool.add_landmark, outputs=[image_display, status_text])
        clear_last_btn.click(tool.clear_last, outputs=[image_display, status_text])
        clear_all_btn.click(tool.clear_all, outputs=[image_display, status_text])
        save_btn.click(tool.save_annotation, outputs=[status_text])
        next_btn.click(tool.next_image, outputs=[image_display, status_text])
        prev_btn.click(tool.prev_image, outputs=[image_display, status_text])

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Annotate facial landmarks on anime characters')
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/raw',
        help='Directory containing images to annotate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/annotated',
        help='Directory to save annotations'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run Gradio server on'
    )

    args = parser.parse_args()

    demo = create_interface(args.image_dir, args.output_dir)
    demo.launch(server_port=args.port, share=False)
