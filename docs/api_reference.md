# APIリファレンス

このドキュメントでは、anime-face-detectorのすべてのパブリックAPIについて詳しく説明します。

## モジュール: anime_face_detector

### create_detector()

検出器インスタンスを作成します。

```python
def create_detector(
    face_detector_name: str = 'yolov3',
    landmark_model_name: str = 'hrnetv2',
    device: str = 'cuda:0',
    flip_test: bool = True,
    box_scale_factor: float = 1.1,
) -> LandmarkDetector
```

#### パラメータ

- **face_detector_name** (`str`, デフォルト: `'yolov3'`)
  - 顔検出に使用するモデル
  - 選択肢: `'yolov3'`, `'faster-rcnn'`
  - `'yolov3'`: 高速、リアルタイムアプリケーション向け
  - `'faster-rcnn'`: 高精度、精度重視の場合に推奨

- **landmark_model_name** (`str`, デフォルト: `'hrnetv2'`)
  - ランドマーク検出に使用するモデル
  - 現在は `'hrnetv2'` のみサポート

- **device** (`str`, デフォルト: `'cuda:0'`)
  - 推論に使用するデバイス
  - 例: `'cuda:0'`, `'cuda:1'`, `'cpu'`
  - CUDA対応GPUがある場合は `'cuda:0'` を推奨

- **flip_test** (`bool`, デフォルト: `True`)
  - ランドマーク検出時に水平反転テストを使用するかどうか
  - `True`: 画像を反転した結果も使用して精度向上（推論時間は約2倍）
  - `False`: 通常の推論のみ（高速）

- **box_scale_factor** (`float`, デフォルト: `1.1`)
  - 顔検出後のバウンディングボックス拡大率
  - 値が大きいほど、顔周辺の広い領域を使用
  - 推奨範囲: 1.0 - 1.3
  - 1.0: 拡大なし、1.1: 10%拡大、1.2: 20%拡大

#### 戻り値

- `LandmarkDetector`: 顔検出とランドマーク検出を行うインスタンス

#### 例外

- `AssertionError`: 無効なモデル名が指定された場合

#### 使用例

```python
from anime_face_detector import create_detector

# デフォルト設定（YOLOv3, CUDA）
detector = create_detector()

# Faster R-CNNを使用、CPUで実行
detector = create_detector(
    face_detector_name='faster-rcnn',
    device='cpu'
)

# flip_testを無効化して高速化
detector = create_detector(
    flip_test=False,
    box_scale_factor=1.2
)
```

---

### get_config_path()

モデルの設定ファイルパスを取得します。

```python
def get_config_path(model_name: str) -> pathlib.Path
```

#### パラメータ

- **model_name** (`str`)
  - モデル名
  - 選択肢: `'yolov3'`, `'faster-rcnn'`, `'hrnetv2'`

#### 戻り値

- `pathlib.Path`: 設定ファイルの絶対パス

#### 例外

- `AssertionError`: 無効なモデル名が指定された場合

#### 使用例

```python
from anime_face_detector import get_config_path

config_path = get_config_path('yolov3')
print(config_path)
# /path/to/anime_face_detector/configs/mmdet/yolov3.py
```

---

### get_checkpoint_path()

モデルのチェックポイントファイルパスを取得します。ファイルが存在しない場合、自動的にダウンロードします。

```python
def get_checkpoint_path(model_name: str) -> pathlib.Path
```

#### パラメータ

- **model_name** (`str`)
  - モデル名
  - 選択肢: `'yolov3'`, `'faster-rcnn'`, `'hrnetv2'`

#### 戻り値

- `pathlib.Path`: チェックポイントファイルの絶対パス

#### 例外

- `AssertionError`: 無効なモデル名が指定された場合

#### 動作

1. `torch.hub.get_dir() + '/checkpoints/'`にファイルがあるか確認
2. ない場合、GitHub Releasesからダウンロード
3. パスを返却

デフォルトのダウンロード先: `~/.cache/torch/hub/checkpoints/`

#### 使用例

```python
from anime_face_detector import get_checkpoint_path

checkpoint_path = get_checkpoint_path('yolov3')
print(checkpoint_path)
# /home/user/.cache/torch/hub/checkpoints/mmdet_anime-face_yolov3.pth
```

---

## クラス: LandmarkDetector

顔検出とランドマーク検出を統合したクラス。

### コンストラクタ

```python
def __init__(
    self,
    landmark_detector_config_or_path: Config | str | pathlib.Path,
    landmark_detector_checkpoint_path: str | pathlib.Path,
    face_detector_config_or_path: Config | str | pathlib.Path | None = None,
    face_detector_checkpoint_path: str | pathlib.Path | None = None,
    device: str = 'cuda:0',
    flip_test: bool = True,
    box_scale_factor: float = 1.1,
)
```

#### パラメータ

- **landmark_detector_config_or_path** (`Config | str | pathlib.Path`)
  - ランドマーク検出器の設定（Configオブジェクトまたはファイルパス）

- **landmark_detector_checkpoint_path** (`str | pathlib.Path`)
  - ランドマーク検出器のチェックポイントファイルパス

- **face_detector_config_or_path** (`Config | str | pathlib.Path | None`, デフォルト: `None`)
  - 顔検出器の設定（Configオブジェクトまたはファイルパス）
  - `None`の場合、顔検出は使用されません

- **face_detector_checkpoint_path** (`str | pathlib.Path | None`, デフォルト: `None`)
  - 顔検出器のチェックポイントファイルパス

- **device** (`str`, デフォルト: `'cuda:0'`)
  - 推論に使用するデバイス

- **flip_test** (`bool`, デフォルト: `True`)
  - ランドマーク検出時に水平反転テストを使用するかどうか

- **box_scale_factor** (`float`, デフォルト: `1.1`)
  - バウンディングボックス拡大率

#### 使用例

```python
from anime_face_detector.detector import LandmarkDetector
from mmengine.config import Config

# 設定ファイルパスを指定
detector = LandmarkDetector(
    landmark_detector_config_or_path='configs/mmpose/hrnetv2.py',
    landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
    face_detector_config_or_path='configs/mmdet/yolov3.py',
    face_detector_checkpoint_path='checkpoints/yolov3.pth',
    device='cuda:0'
)

# Configオブジェクトを直接使用
landmark_config = Config.fromfile('configs/mmpose/hrnetv2.py')
face_config = Config.fromfile('configs/mmdet/yolov3.py')

detector = LandmarkDetector(
    landmark_detector_config_or_path=landmark_config,
    landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
    face_detector_config_or_path=face_config,
    face_detector_checkpoint_path='checkpoints/yolov3.pth'
)
```

---

### \_\_call\_\_()

画像から顔を検出し、ランドマークを推定します。

```python
def __call__(
    self,
    image_or_path: np.ndarray | str | pathlib.Path,
    boxes: list[np.ndarray] | None = None,
) -> list[dict[str, np.ndarray]]
```

#### パラメータ

- **image_or_path** (`np.ndarray | str | pathlib.Path`)
  - 入力画像
  - `np.ndarray`: BGR形式の画像配列（OpenCV形式）
  - `str` または `pathlib.Path`: 画像ファイルのパス

- **boxes** (`list[np.ndarray] | None`, デフォルト: `None`)
  - 顔のバウンディングボックスのリスト
  - 各ボックスは `[x0, y0, x1, y1]` または `[x0, y0, x1, y1, score]` 形式
  - `None`の場合、顔検出器を使用して自動検出
  - 顔検出器が指定されていない場合、画像全体を顔領域として使用

#### 戻り値

- `list[dict[str, np.ndarray]]`: 検出結果のリスト

各要素は以下の構造を持つ辞書:

```python
{
    'bbox': np.ndarray,      # shape: (5,), [x0, y0, x1, y1, score]
    'keypoints': np.ndarray  # shape: (28, 3), [[x, y, score], ...]
}
```

- `bbox`: 顔のバウンディングボックス
  - `x0, y0`: 左上座標
  - `x1, y1`: 右下座標
  - `score`: 検出信頼度 (0.0 - 1.0)

- `keypoints`: 28個のランドマーク座標
  - 各ランドマークは `[x, y, score]` の形式
  - `x, y`: ピクセル座標
  - `score`: キーポイント信頼度 (0.0 - 1.0)

#### ランドマークインデックス

```
0-2:   左側の顔の輪郭
3-5:   右側の顔の輪郭
5-7:   左眉
8-10:  右眉
11-13: 左目
14-16: 鼻
17-19: 右目
20-22: 口（右側）
23:    口の中央下
24-26: 口の下部
27:    口の中央
```

#### 使用例

```python
import cv2
from anime_face_detector import create_detector

detector = create_detector('yolov3')

# 画像ファイルパスから検出
preds = detector('path/to/image.jpg')

# NumPy配列から検出
image = cv2.imread('path/to/image.jpg')
preds = detector(image)

# 事前に検出した顔ボックスを使用
boxes = [np.array([100, 100, 300, 300])]
preds = detector(image, boxes=boxes)

# 結果の利用
for pred in preds:
    bbox = pred['bbox']
    keypoints = pred['keypoints']

    print(f"顔の位置: ({bbox[0]:.1f}, {bbox[1]:.1f}) - ({bbox[2]:.1f}, {bbox[3]:.1f})")
    print(f"検出信頼度: {bbox[4]:.3f}")

    for i, (x, y, score) in enumerate(keypoints):
        print(f"ランドマーク{i}: ({x:.1f}, {y:.1f}), 信頼度: {score:.3f}")
```

#### 画像への描画例

```python
import cv2
import numpy as np
from anime_face_detector import create_detector

detector = create_detector('yolov3')
image = cv2.imread('input.jpg')
preds = detector(image)

# バウンディングボックスの描画
for pred in preds:
    bbox = pred['bbox'].astype(int)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # ランドマークの描画
    keypoints = pred['keypoints']
    for x, y, score in keypoints:
        if score > 0.5:  # 信頼度0.5以上のみ描画
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

cv2.imwrite('output.jpg', image)
```

---

## データ形式

### バウンディングボックス形式

```python
bbox: np.ndarray  # shape: (5,), dtype: float32
# [x0, y0, x1, y1, score]
```

- `x0, y0`: 左上のピクセル座標
- `x1, y1`: 右下のピクセル座標
- `score`: 検出信頼度 (0.0 - 1.0)

### キーポイント形式

```python
keypoints: np.ndarray  # shape: (28, 3), dtype: float32
# [[x0, y0, score0],
#  [x1, y1, score1],
#  ...
#  [x27, y27, score27]]
```

- `x, y`: キーポイントのピクセル座標
- `score`: キーポイントの信頼度 (0.0 - 1.0)

---

## 内部メソッド（参考用）

以下のメソッドは内部実装であり、通常は直接呼び出す必要はありません。

### _detect_faces()

```python
def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]
```

mmdetを使用して顔を検出します。

### _detect_landmarks()

```python
def _detect_landmarks(
    self, image: np.ndarray, boxes: list[np.ndarray]
) -> list[dict[str, np.ndarray]]
```

mmposeを使用してランドマークを検出します。

### _load_image()

```python
@staticmethod
def _load_image(image_or_path: np.ndarray | str | pathlib.Path) -> np.ndarray
```

画像を読み込み、NumPy配列に変換します。

### _update_pred_box()

```python
def _update_pred_box(self, pred_boxes: np.ndarray) -> list[np.ndarray]
```

バウンディングボックスを`box_scale_factor`で拡大します。

---

## 高度な使用例

### カスタム設定での初期化

```python
from anime_face_detector.detector import LandmarkDetector
from mmengine.config import Config

# カスタム設定を作成
landmark_config = Config.fromfile('configs/mmpose/hrnetv2.py')
landmark_config.model.test_cfg.flip_test = False  # flip_testを無効化

detector = LandmarkDetector(
    landmark_detector_config_or_path=landmark_config,
    landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
    device='cuda:0'
)
```

### バッチ処理

```python
import glob
from anime_face_detector import create_detector

detector = create_detector('yolov3')

image_paths = glob.glob('images/*.jpg')
for image_path in image_paths:
    preds = detector(image_path)
    # 処理...
```

### 信頼度でのフィルタリング

```python
from anime_face_detector import create_detector

detector = create_detector('yolov3')
preds = detector('image.jpg')

# 信頼度0.9以上の顔のみ使用
high_conf_preds = [pred for pred in preds if pred['bbox'][4] > 0.9]

# 信頼度0.5以上のランドマークのみ使用
for pred in preds:
    keypoints = pred['keypoints']
    reliable_keypoints = keypoints[keypoints[:, 2] > 0.5]
```

---

## エラーとトラブルシューティング

### よくあるエラー

#### AssertionError: Invalid model name

```python
detector = create_detector('invalid-model')
# AssertionError: assertion failed
```

解決策: サポートされているモデル名を使用してください（`'yolov3'`, `'faster-rcnn'`, `'hrnetv2'`）。

#### CUDA out of memory

```python
# RuntimeError: CUDA out of memory
```

解決策:
- より小さい画像サイズを使用
- `device='cpu'`でCPUモードに切り替え
- 他のCUDAプロセスを終了

#### KeyError: 'xxx is not in the xxx registry'

レジストリのスコープエラー。通常は内部で自動処理されますが、カスタム実装時に発生する可能性があります。

解決策: `DefaultScope.overwrite_default_scope()`を適切に使用してください。

---

## 型ヒント

```python
from typing import Union
import pathlib
import numpy as np
from anime_face_detector import LandmarkDetector

ImageInput = Union[np.ndarray, str, pathlib.Path]
BoundingBox = np.ndarray  # shape: (5,)
Keypoints = np.ndarray    # shape: (28, 3)
Detection = dict[str, np.ndarray]  # {'bbox': BoundingBox, 'keypoints': Keypoints}
```
