# 開発ガイド

このドキュメントでは、anime-face-detectorの開発環境のセットアップ、コーディング規約、テスト方法、コントリビューション方法について説明します。

## 開発環境のセットアップ

### 前提条件

- Python 3.10 または 3.11
- CUDA 11.8 以降（GPU使用時）
- Git
- uv（推奨）または pip

### インストール手順

#### 1. リポジトリのクローン

```bash
git clone https://github.com/hysts/anime-face-detector.git
cd anime-face-detector
```

#### 2. 依存関係のインストール（uv使用）

```bash
# システム依存関係
sudo apt-get install -y ninja-build

# 仮想環境の作成
uv venv .venv && uv sync
uv pip install wheel

# xtcocoapiのインストール（mmpose依存）
mkdir -p deps && cd deps
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi && ../../.venv/bin/python -m pip install -e . && cd ../..

# PyTorch（CUDA 12.8対応）
# 他のCUDAバージョンの場合: https://pytorch.org/get-started/previous-versions/
uv pip install torch==2.9.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

# OpenMMLab依存関係
uv pip install openmim mmengine

# mmcv（CUDA演算子込み）
# GPU別のアーキテクチャ:
# - RTX 50XX (Blackwell): "12.0"
# - H100 (Hopper): "9.0"
# - RTX 40XX (Ada): "8.9"
# - RTX 30XX (Ampere): "8.0,8.6"
# - RTX 20XX (Turing): "7.5"
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir --no-build-isolation

# mmdet, mmpose
uv pip install --no-cache-dir mmdet==3.2.0 mmpose==1.3.2

# 開発用依存関係
uv pip install --no-cache-dir gradio ruff pre-commit
```

#### 3. エディタブルモードでのインストール

```bash
pip install -e .
```

これにより、ソースコードの変更がすぐに反映されます。

#### 4. pre-commitフックのセットアップ

```bash
pre-commit install
```

---

## プロジェクト構造

```
anime-face-detector/
├── anime_face_detector/          # メインパッケージ
│   ├── __init__.py               # パブリックAPI
│   ├── detector.py               # LandmarkDetectorクラス
│   └── configs/                  # モデル設定
│       ├── mmdet/
│       │   ├── yolov3.py
│       │   └── faster-rcnn.py
│       └── mmpose/
│           └── hrnetv2.py
├── docs/                         # ドキュメント
│   ├── architecture.md
│   ├── api_reference.md
│   ├── models_and_configs.md
│   └── development.md
├── assets/                       # デモ画像
├── demo_gradio.py                # Gradioデモ
├── demo.ipynb                    # Jupyterデモ
├── pyproject.toml                # プロジェクト設定
├── setup.py                      # セットアップスクリプト
├── .pre-commit-config.yaml       # pre-commit設定
├── CLAUDE.md                     # Claude Code用ガイド
└── README.md                     # READMEファイル
```

---

## コーディング規約

### スタイルガイド

このプロジェクトは **ruff** を使用してコードの品質を維持しています。

#### ruffの設定（pyproject.toml）

```toml
[tool.ruff]
line-length = 88
indent-width = 4
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = []

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
line-ending = "lf"
```

### コーディングスタイル

- **文字列**: シングルクォート（`'text'`）を使用
- **行の長さ**: 最大88文字
- **インデント**: スペース4つ
- **改行コード**: LF（Unix形式）
- **import順序**: 標準ライブラリ → サードパーティ → ローカル

### ruffの使用

```bash
# リントチェック
ruff check .

# 自動修正
ruff check . --fix

# フォーマット
ruff format .

# すべてを実行
ruff check . --fix && ruff format .
```

### pre-commit

コミット前に自動でruffが実行されます。

```bash
# 手動実行
pre-commit run --all-files

# 特定のフックのみ実行
pre-commit run ruff --all-files
```

---

## 型ヒント

Python 3.10+の型ヒントを積極的に使用してください。

### 推奨される型ヒント

```python
from __future__ import annotations

import pathlib
from typing import Union

import numpy as np
from mmengine.config import Config


def example_function(
    image_path: str | pathlib.Path,
    config: Config | None = None,
    threshold: float = 0.5,
) -> list[dict[str, np.ndarray]]:
    """関数の説明.

    Args:
        image_path: 画像ファイルのパス
        config: 設定オブジェクト（オプション）
        threshold: 閾値（デフォルト: 0.5）

    Returns:
        検出結果のリスト
    """
    pass
```

### 型チェック

現在は型チェッカー（mypy等）を使用していませんが、将来的な導入を検討しています。

---

## テスト

### 現在の状態

現在、自動テストは実装されていません。手動テストを行ってください。

### 手動テスト

#### 基本的な動作確認

```python
from anime_face_detector import create_detector
import cv2

# YOLOv3でテスト
detector = create_detector('yolov3', device='cpu')
image = cv2.imread('assets/input.jpg')
preds = detector(image)

assert len(preds) > 0, '顔が検出されませんでした'
assert preds[0]['keypoints'].shape == (28, 3), 'ランドマーク数が不正です'
print('テスト成功: YOLOv3')

# Faster R-CNNでテスト
detector = create_detector('faster-rcnn', device='cpu')
preds = detector(image)

assert len(preds) > 0, '顔が検出されませんでした'
print('テスト成功: Faster R-CNN')
```

#### Gradioデモでのテスト

```bash
python demo_gradio.py --device cpu
```

ブラウザで http://localhost:7860 を開いて、UIから動作確認を行います。

### 将来のテスト実装（TODO）

```bash
# pytestの導入予定
pip install pytest pytest-cov

# テスト実行
pytest tests/

# カバレッジ測定
pytest --cov=anime_face_detector tests/
```

---

## デバッグ

### ログ出力

OpenMMLabのロギング機能を使用できます。

```python
from mmengine.logging import print_log

print_log('デバッグメッセージ', logger='current', level='DEBUG')
```

### よくあるデバッグポイント

#### 1. モデルが正しく読み込まれているか

```python
detector = create_detector('yolov3')
print(f'Face detector: {detector.face_detector}')
print(f'Landmark detector: {detector.landmark_detector}')
```

#### 2. 設定が正しく適用されているか

```python
from mmengine.config import Config

config = Config.fromfile('anime_face_detector/configs/mmdet/yolov3.py')
print(config.pretty_text)
```

#### 3. デバイスが正しく設定されているか

```python
import torch

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
```

#### 4. 推論時のスコープ確認

```python
from mmengine.registry import DefaultScope

print(f'Current scope: {DefaultScope.get_current_instance().scope_name}')
```

---

## 新機能の追加

### 新しい顔検出器の追加

1. **設定ファイルの作成**

```bash
touch anime_face_detector/configs/mmdet/new_detector.py
```

2. **設定ファイルの記述**

```python
# anime_face_detector/configs/mmdet/new_detector.py
model = dict(
    type='YourDetectorType',
    # ... 設定
)

test_pipeline = [
    # ... パイプライン
]

test_dataloader = dict(
    # ... データローダー
)
```

3. **__init__.pyの更新**

```python
# anime_face_detector/__init__.py
def get_config_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'new_detector', 'hrnetv2']  # 追加
    # ...

def get_checkpoint_path(model_name: str) -> pathlib.Path:
    assert model_name in ['faster-rcnn', 'yolov3', 'new_detector', 'hrnetv2']  # 追加
    # ...

def create_detector(
    face_detector_name: str = 'yolov3',
    # ...
) -> LandmarkDetector:
    assert face_detector_name in ['yolov3', 'faster-rcnn', 'new_detector']  # 追加
    # ...
```

4. **チェックポイントの配置**

```bash
# GitHub Releasesにアップロードするか、ローカルに配置
cp your_model.pth ~/.cache/torch/hub/checkpoints/mmdet_anime-face_new_detector.pth
```

### 新しいランドマークモデルの追加

同様の手順で、`configs/mmpose/`に設定ファイルを追加します。

---

## ドキュメントの更新

### ドキュメントの構成

- [architecture.md](architecture.md): アーキテクチャの解説
- [api_reference.md](api_reference.md): API仕様
- [models_and_configs.md](models_and_configs.md): モデルと設定の詳細
- [development.md](development.md): このファイル

### ドキュメントの記述ルール

- マークダウン形式
- コード例を豊富に含める
- 日本語で記述（国際化は将来の課題）
- 見出しレベルを適切に使用

---

## リリースプロセス

### バージョニング

セマンティックバージョニング（SemVer）に従います。

```
MAJOR.MINOR.PATCH
例: 1.2.3
```

- **MAJOR**: 互換性のない変更
- **MINOR**: 後方互換性のある機能追加
- **PATCH**: 後方互換性のあるバグ修正

### リリース手順（メンテナー向け）

1. **バージョン番号の更新**

```bash
# pyproject.toml
version = "0.1.0"

# setup.py
version='0.1.0'
```

2. **CHANGELOG.mdの更新**

```markdown
## [0.1.0] - 2026-01-20

### Added
- 新機能の説明

### Changed
- 変更内容

### Fixed
- 修正内容
```

3. **タグの作成**

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

4. **PyPIへのアップロード**

```bash
python -m build
python -m twine upload dist/*
```

5. **GitHub Releasesの作成**

- リリースノートを記述
- モデルファイルをアップロード

---

## Dockerイメージのビルド

### ローカルでのビルド

```bash
docker build -t anime-face-detector:latest .
```

### マルチステージビルド

Dockerfileは最適化されたマルチステージビルドを使用しています。

```dockerfile
# ビルドステージ
FROM python:3.11-slim as builder
# ... 依存関係のインストール

# ランタイムステージ
FROM python:3.11-slim
# ... 必要なファイルのみコピー
```

### GitHub Container Registryへのプッシュ

```bash
docker tag anime-face-detector:latest ghcr.io/username/anime-face-detector:latest
docker push ghcr.io/username/anime-face-detector:latest
```

---

## コントリビューション

### Issue報告

バグや機能要望は、GitHubのIssueで報告してください。

#### バグレポートのテンプレート

```markdown
## バグの説明
簡潔にバグを説明してください。

## 再現手順
1. xxx を実行
2. yyy を確認
3. エラー発生

## 期待される動作
本来はこうあるべきという動作

## 実際の動作
実際に起きている動作

## 環境
- OS: Ubuntu 22.04
- Python: 3.11
- CUDA: 12.8
- anime-face-detector: 0.1.0
```

### Pull Request

1. **Forkとクローン**

```bash
git clone https://github.com/your-username/anime-face-detector.git
cd anime-face-detector
git remote add upstream https://github.com/hysts/anime-face-detector.git
```

2. **ブランチの作成**

```bash
git checkout -b feature/my-new-feature
```

3. **変更の実装**

- コーディング規約に従う
- pre-commitフックをパス
- ドキュメントを更新

4. **コミット**

```bash
git add .
git commit -m "Add my new feature"
```

コミットメッセージの形式:

```
<type>: <subject>

<body>

<footer>
```

タイプ:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: コードスタイル
- `refactor`: リファクタリング
- `test`: テスト
- `chore`: その他

例:
```
feat: Add support for custom landmark models

- Allow users to specify custom config and checkpoint
- Update documentation with usage examples

Closes #42
```

5. **プッシュ**

```bash
git push origin feature/my-new-feature
```

6. **Pull Requestの作成**

GitHubでPull Requestを作成し、変更内容を説明します。

### レビュープロセス

- メンテナーがコードレビューを行います
- 必要に応じて修正をリクエストします
- 承認後、マージされます

---

## トラブルシューティング

### よくある開発環境の問題

#### mmcvのビルドエラー

```
error: cannot find -lcudart
```

解決策: CUDA_HOMEを設定してください。

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### pre-commitフックの失敗

```
Ruff...........Failed
```

解決策: ruffで自動修正してから再コミット。

```bash
ruff check . --fix
ruff format .
git add .
git commit -m "Fix linting issues"
```

#### Gradioデモが起動しない

```
ModuleNotFoundError: No module named 'gradio'
```

解決策: Gradioをインストール。

```bash
pip install gradio
```

---

## パフォーマンスプロファイリング

### 推論速度の測定

```python
import time
from anime_face_detector import create_detector
import cv2

detector = create_detector('yolov3', device='cuda:0')
image = cv2.imread('assets/input.jpg')

# ウォームアップ
for _ in range(10):
    detector(image)

# 計測
times = []
for _ in range(100):
    start = time.perf_counter()
    preds = detector(image)
    end = time.perf_counter()
    times.append(end - start)

import numpy as np
print(f'平均推論時間: {np.mean(times)*1000:.2f} ms')
print(f'標準偏差: {np.std(times)*1000:.2f} ms')
print(f'FPS: {1/np.mean(times):.2f}')
```

### メモリ使用量の測定

```python
import torch
from anime_face_detector import create_detector

torch.cuda.reset_peak_memory_stats()

detector = create_detector('yolov3', device='cuda:0')
image = cv2.imread('assets/input.jpg')
preds = detector(image)

max_memory = torch.cuda.max_memory_allocated() / 1024**2
print(f'最大メモリ使用量: {max_memory:.2f} MB')
```

---

## その他のリソース

### 公式ドキュメント

- [mmdetection docs](https://mmdetection.readthedocs.io/)
- [mmpose docs](https://mmpose.readthedocs.io/)
- [mmengine docs](https://mmengine.readthedocs.io/)
- [mmcv docs](https://mmcv.readthedocs.io/)

### コミュニティ

- GitHub Issues: バグ報告、機能要望
- GitHub Discussions: 質問、アイデア共有

### 関連プロジェクト

- [OpenMMLab](https://github.com/open-mmlab)
- [anime-face-detector original](https://github.com/hysts/anime-face-detector)

---

## まとめ

このドキュメントでは、anime-face-detectorの開発環境のセットアップからコントリビューションまでを解説しました。不明な点があれば、GitHubのIssueやDiscussionsで質問してください。

Happy coding!
