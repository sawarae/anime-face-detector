# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

アニメ顔検出・ランドマーク検出ライブラリ（Pure PyTorch実装）。YOLOv8（顔検出）とHRNetV2（28点ランドマーク検出）を使用。

**重要**: このバージョンはmmdetection/mmpose/mmcv依存を完全に削除し、Pure PyTorchで実装されています。

## 依存関係

- Python 3.10-3.11
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python-headless >= 4.5.4.58
- ultralytics >= 8.0.0 (YOLOv8)
- huggingface-hub >= 0.20.0

**旧版の依存関係（削除済み）**:
- ❌ mmengine
- ❌ mmcv
- ❌ mmdet
- ❌ mmpose

## インストール

```bash
pip install -e .
# または
pip install torch torchvision ultralytics huggingface-hub opencv-python-headless numpy
```

## 開発コマンド

### pre-commitフック
```bash
pre-commit install
pre-commit run --all-files
```

### テスト
```bash
python test_new_implementation.py
```

### Gradioデモ
```bash
# Gradioをインストール（Gradio 5.x以上が必要）
pip install "gradio>=5.0.0"
# または uv を使用
uv pip install "gradio>=5.0.0"

# CUDAが利用可能な場合は自動的にGPUを使用、それ以外はCPUを使用
python demo_gradio.py

# 明示的にデバイスを指定する場合
python demo_gradio.py --device cpu
python demo_gradio.py --device cuda:0
```

## アーキテクチャ

### 2段階検出パイプライン
1. **顔検出**: Ultralytics YOLOv8で顔のバウンディングボックスを検出
2. **ランドマーク検出**: Pure PyTorch HRNetV2で28点のランドマークを検出

### 主要コンポーネント

- `anime_face_detector/__init__.py`: エントリーポイント。`create_detector()`で検出器を生成
- `anime_face_detector/detector.py`: `LandmarkDetector`クラス。YOLOv8とHRNetV2を統合
- `anime_face_detector/models/hrnetv2.py`: HRNetV2モデルの実装（Pure PyTorch）
- `anime_face_detector/models/inference.py`: ランドマーク検出の推論ユーティリティ

### モデル自動ダウンロード

#### YOLOv8顔検出モデル
- デフォルト: `face_yolov8n.pt` (Hugging Face: Bingsu/adetailer)
- 初回実行時に自動ダウンロード

#### HRNetV2ランドマーク検出モデル
- ファイル名: `mmpose_anime-face_hrnetv2.pth`
- ソース: https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/
- 初回実行時に自動ダウンロード

### Pure PyTorch実装の詳細

#### HRNetV2アーキテクチャ
- **Backbone**: 4段階のマルチスケールブランチ
  - Stage 1: Bottleneck blocks (64チャンネル)
  - Stage 2: 2ブランチ (18, 36チャンネル)
  - Stage 3: 3ブランチ (18, 36, 72チャンネル)
  - Stage 4: 4ブランチ (18, 36, 72, 144チャンネル)
- **Head**: 全ブランチを連結 (270チャンネル) → ヒートマップ (28チャンネル)
- **入力サイズ**: 256x256
- **出力**: 28点のキーポイントヒートマップ (64x64)

#### 推論パイプライン
1. YOLOv8で顔領域を検出
2. 各顔領域をクロップ・アフィン変換 (256x256)
3. HRNetV2でヒートマップを生成
4. argmaxでキーポイント座標を抽出
5. 逆アフィン変換で元画像座標系に戻す

### 使用例

```python
from anime_face_detector import create_detector
import cv2

# デフォルトモデルを使用（自動ダウンロード）
detector = create_detector(device='cuda:0')

# カスタムYOLOv8モデルを使用
detector = create_detector(
    face_detector_checkpoint_path='path/to/custom_yolov8.pt',
    device='cuda:0'
)

# 推論実行
image = cv2.imread('image.jpg')  # BGR形式
preds = detector(image)

# 結果の取得
for pred in preds:
    bbox = pred['bbox']  # [x0, y0, x1, y1, score]
    keypoints = pred['keypoints']  # [[x, y, score], ...] (28点)
```

## 28点ランドマーク構成

- 0-4: 顔輪郭
- 5-10: 眉
- 11-13, 17-19: 目
- 14-16: 鼻
- 20-27: 口

## コードスタイル

- フォーマッター/リンター: ruff
- 文字列: シングルクォート
- 行の長さ: 88
- 改行: LF

### ruffコマンド
```bash
ruff check .        # リント
ruff format .       # フォーマット
```

## 移行ガイド（旧版から）

### 主な変更点

1. **API変更**:
   ```python
   # 旧版
   detector = create_detector('yolov3', device='cuda:0')

   # 新版
   detector = create_detector(device='cuda:0')  # デフォルトでYOLOv8を使用
   ```

2. **依存関係**: mmdetection/mmpose不要

3. **設定ファイル**: 不要（モデルに組み込み済み）

4. **返り値の形式**: 変更なし（互換性あり）
