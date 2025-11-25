# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

アニメ顔検出・ランドマーク検出ライブラリ。mmdetection（顔検出）とmmpose（28点ランドマーク検出）をベースに構築されている。

## インストール

```bash
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmpose
pip install -e .
```

## 開発コマンド

### pre-commitフック
```bash
pre-commit install
pre-commit run --all-files
```

### Gradioデモの実行
```bash
pip install gradio
python demo_gradio.py --device cpu  # CPUで実行
python demo_gradio.py --detector yolov3  # デフォルト
python demo_gradio.py --detector faster-rcnn
```

## アーキテクチャ

### 2段階検出パイプライン
1. **顔検出** (mmdet): YOLOv3またはFaster R-CNNで顔のバウンディングボックスを検出
2. **ランドマーク検出** (mmpose): HRNetV2で28点のランドマークを検出

### 主要コンポーネント

- `anime_face_detector/__init__.py`: エントリーポイント。`create_detector()`で検出器を生成。モデルは初回実行時にGitHub Releasesから自動ダウンロード
- `anime_face_detector/detector.py`: `LandmarkDetector`クラス。顔検出とランドマーク検出を統合
- `anime_face_detector/configs/mmdet/`: YOLOv3、Faster R-CNNの設定ファイル
- `anime_face_detector/configs/mmpose/hrnetv2.py`: 28点ランドマーク検出の設定

### 使用例
```python
from anime_face_detector import create_detector
import cv2

detector = create_detector('yolov3', device='cuda:0')  # または 'faster-rcnn'
image = cv2.imread('image.jpg')  # BGR形式
preds = detector(image)
# preds[i]['bbox']: [x0, y0, x1, y1, score]
# preds[i]['keypoints']: [[x, y, score], ...] (28点)
```

## コードスタイル

- フォーマッター: yapf
- import順序: isort
- 型チェック: mypy（`--ignore-missing-imports`）
- 文字列: ダブルクォート
- 改行: LF
