import argparse
import pathlib

import cv2
import gradio as gr
import numpy as np
import PIL.Image
import torch

import anime_face_detector


def create_detect_fn(detector: anime_face_detector.LandmarkDetector):
    """Create a detection function with the detector bound via closure."""

    def detect(
        img, face_score_threshold: float, landmark_score_threshold: float
    ) -> PIL.Image.Image:
        image = cv2.imread(img)
        preds = detector(image)

        res = image.copy()
        for pred in preds:
            box = pred['bbox']
            box, score = box[:4], box[4]
            if score < face_score_threshold:
                continue
            box = np.round(box).astype(int)

            lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

            cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

            pred_pts = pred['keypoints']
            for *pt, score in pred_pts:
                if score < landmark_score_threshold:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                pt = np.round(pt).astype(int)
                cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        image_pil = PIL.Image.fromarray(res)
        return image_pil

    return detect


def main():
    parser = argparse.ArgumentParser()
    # Auto-detect device: use CUDA if available, otherwise CPU
    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device',
                        type=str,
                        default=default_device,
                        choices=['cuda:0', 'cpu'])
    parser.add_argument('--face-score-threshold', type=float, default=0.5)
    parser.add_argument('--landmark-score-threshold', type=float, default=0.3)
    parser.add_argument('--score-slider-step', type=float, default=0.05)
    parser.add_argument('--port', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    sample_path = pathlib.Path('input.jpg')
    if not sample_path.exists():
        torch.hub.download_url_to_file(
            'https://raw.githubusercontent.com/hysts/anime-face-detector/main/assets/input.jpg',
            sample_path.as_posix())

    # Use default YOLOv8 detector (auto-downloads face_yolov8n.pt)
    detector = anime_face_detector.create_detector(device=args.device)
    detect_fn = create_detect_fn(detector)

    title = 'hysts/anime-face-detector'
    description = 'Demo for hysts/anime-face-detector. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.'
    article = "<a href='https://github.com/hysts/anime-face-detector'>GitHub Repo</a>"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f'# {title}')
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type='filepath', label='Input')
                face_score_slider = gr.Slider(
                    0,
                    1,
                    step=args.score_slider_step,
                    value=args.face_score_threshold,
                    label='Face Score Threshold',
                )
                landmark_score_slider = gr.Slider(
                    0,
                    1,
                    step=args.score_slider_step,
                    value=args.landmark_score_threshold,
                    label='Landmark Score Threshold',
                )
                submit_btn = gr.Button('Detect', variant='primary')

            with gr.Column():
                output_image = gr.Image(type='pil', label='Output')

        gr.Examples(
            examples=[
                [
                    sample_path.as_posix(),
                    args.face_score_threshold,
                    args.landmark_score_threshold,
                ],
            ],
            inputs=[input_image, face_score_slider, landmark_score_slider],
            outputs=output_image,
            fn=detect_fn,
        )

        submit_btn.click(
            fn=detect_fn,
            inputs=[input_image, face_score_slider, landmark_score_slider],
            outputs=output_image,
        )

        gr.Markdown(article)

    demo.launch(
        debug=args.debug,
        share=args.share,
        server_port=args.port,
        server_name='0.0.0.0',
    )


if __name__ == '__main__':
    main()
