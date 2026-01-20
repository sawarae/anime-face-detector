# アーキテクチャ

このドキュメントでは、anime-face-detectorのアーキテクチャと内部実装について詳しく解説します。

## 概要

anime-face-detectorは、OpenMMLabエコシステム（mmdetection, mmpose）をベースとした2段階の検出パイプラインで構成されています。

```
入力画像 → 顔検出 (mmdet) → ランドマーク検出 (mmpose) → 結果出力
```

## システム構成

### 依存関係

本ライブラリは、OpenMMLab 2.0のフレームワーク上に構築されています。

```
anime-face-detector
├── mmengine (0.10.7)    # 設定管理、レジストリ、ロギング
├── mmcv (2.1.0)         # 画像処理、CUDA演算子
├── mmdet (3.2.0)        # 物体検出フレームワーク
└── mmpose (1.3.2)       # 姿勢推定・ランドマーク検出フレームワーク
```

### ディレクトリ構造

```
anime_face_detector/
├── __init__.py              # エントリーポイント（create_detector, get_config_path）
├── detector.py              # LandmarkDetectorクラス（メインロジック）
└── configs/
    ├── mmdet/               # 顔検出モデル設定
    │   ├── yolov3.py        # YOLOv3設定
    │   └── faster-rcnn.py   # Faster R-CNN設定
    └── mmpose/              # ランドマーク検出モデル設定
        └── hrnetv2.py       # HRNetV2設定（28点ランドマーク）
```

## 2段階検出パイプライン

### ステージ1: 顔検出 (mmdetection)

顔検出では、画像から顔領域のバウンディングボックスを検出します。

#### サポートされる検出器

1. **YOLOv3**（デフォルト）
   - 高速な1段階検出器
   - リアルタイムアプリケーションに適している
   - モデルサイズ: 約120MB

2. **Faster R-CNN**
   - 高精度な2段階検出器
   - 検出精度を重視する場合に推奨
   - モデルサイズ: 約160MB

#### 検出フロー

```python
# detector.py:_detect_faces()
1. DefaultScope.overwrite_default_scope('mmdet')でスコープを設定
2. inference_detector()でmmdet 3.x APIを使用して検出
3. DetDataSampleからbboxes, scoresを抽出
4. [x0, y0, x1, y1, score]形式に変換
5. box_scale_factorでボックスを拡大（デフォルト1.1倍）
```

バウンディングボックスの拡大により、ランドマーク検出に必要な顔周辺の文脈情報を確保します。

### ステージ2: ランドマーク検出 (mmpose)

検出された顔領域から、28点のランドマークを検出します。

#### HRNetV2アーキテクチャ

- **バックボーン**: HRNetV2 (High-Resolution Network v2)
- **特徴**: 高解像度表現を維持しながらマルチスケール特徴を学習
- **出力**: 28個のキーポイント座標と信頼度スコア

#### ランドマーク配置

28点のランドマークは以下の領域をカバーします。

```
- 顔の輪郭: 6点 (左側: 0-2, 右側: 3-5)
- 左眉: 3点 (5-7)
- 右眉: 3点 (8-10)
- 左目: 3点 (11-13)
- 右目: 3点 (17-19)
- 鼻: 3点 (14-16)
- 口: 7点 (23-27, 20-22)
```

#### 検出フロー

```python
# detector.py:_detect_landmarks()
1. DefaultScope.overwrite_default_scope('mmpose')でスコープを設定
2. bbox座標を(N, 4)のxyxy形式に変換
3. inference_topdown()でmmpose 1.x APIを使用して検出
4. PoseDataSampleからkeypoints, keypoint_scoresを抽出
5. [x, y, score]形式に変換して返却
```

## OpenMMLab 2.0との統合

### スコープ管理

mmdetとmmposeは独自のレジストリを持つため、スコープの切り替えが必要です。

```python
# mmdetでの検出
with DefaultScope.overwrite_default_scope('mmdet'):
    result = inference_detector(self.face_detector, image)

# mmposeでの検出
with DefaultScope.overwrite_default_scope('mmpose'):
    results = inference_topdown(self.landmark_detector, image, bboxes)
```

### API変更への対応

#### mmdet 3.x

- 旧: `inference_detector()`が`numpy.ndarray`を返す
- 新: `inference_detector()`が`DetDataSample`を返す

```python
# mmdet 3.x
result = inference_detector(model, image)
bboxes = result.pred_instances.bboxes.cpu().numpy()
scores = result.pred_instances.scores.cpu().numpy()
```

#### mmpose 1.x

- 旧: `inference_top_down_pose_model()`を使用
- 新: `inference_topdown()`を使用、`PoseDataSample`を返す

```python
# mmpose 1.x
results = inference_topdown(model, image, bboxes, bbox_format='xyxy')
keypoints = results[0].pred_instances.keypoints[0]
scores = results[0].pred_instances.keypoint_scores[0]
```

### 設定管理

mmengineの`Config`クラスを使用して設定を読み込みます。

```python
from mmengine.config import Config

config = Config.fromfile('configs/mmdet/yolov3.py')
```

設定ファイルはPython形式で記述され、継承や変数定義をサポートします。

## モデルの初期化

### チェックポイントの自動ダウンロード

モデルファイルは初回実行時に自動的にダウンロードされます。

```python
# __init__.py:get_checkpoint_path()
model_dir = pathlib.Path(torch.hub.get_dir()) / 'checkpoints'
# デフォルト: ~/.cache/torch/hub/checkpoints/

if not model_path.exists():
    url = f'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/{file_name}'
    torch.hub.download_url_to_file(url, model_path.as_posix())
```

ダウンロード済みのモデルは再利用されます。

### dataset_metaの設定

mmposeモデルには、データセット固有のメタ情報が必要です。

```python
# detector.py:_init_pose_model()
dataset_meta = {
    'dataset_name': 'anime_face',
    'num_keypoints': 28,
    'keypoint_info': {...},
    'skeleton_info': {...},
    'joint_weights': [1.0] * 28,
    'sigmas': [0.025] * 28,
    'flip_indices': [...],
}
model.dataset_meta = dataset_meta
```

これにより、推論時にデータセット情報が正しく参照されます。

### test_cfgの設定

flip_testなどのテスト時の設定を動的に適用します。

```python
# flip_testを有効化
model.test_cfg['flip_test'] = True
model.cfg.model.test_cfg['flip_test'] = True
```

flip_testを有効にすると、画像を水平反転した結果も使用して精度を向上させます。

## 推論処理フロー

### 入力処理

```python
def __call__(self, image_or_path, boxes=None):
    # 1. 画像読み込み（BGR形式）
    image = self._load_image(image_or_path)

    # 2. 顔検出（boxesが未指定の場合）
    if boxes is None:
        if self.face_detector is not None:
            boxes = self._detect_faces(image)
        else:
            # 顔検出器がない場合は画像全体を使用
            h, w = image.shape[:2]
            boxes = [np.array([0, 0, w - 1, h - 1, 1])]

    # 3. ランドマーク検出
    return self._detect_landmarks(image, boxes)
```

### 出力形式

```python
[
    {
        'bbox': np.array([x0, y0, x1, y1, score]),  # 顔のバウンディングボックス
        'keypoints': np.array([
            [x, y, score],  # ランドマーク0
            [x, y, score],  # ランドマーク1
            ...
            [x, y, score],  # ランドマーク27
        ])
    },
    ...  # 検出された顔の数だけ繰り返し
]
```

## パフォーマンス最適化

### CUDA最適化

mmcvは、CUDA演算子を含むようにビルドすることで高速化できます。

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0
```

`TORCH_CUDA_ARCH_LIST`は、使用するGPUアーキテクチャに合わせて設定します。

### デバイス選択

```python
# CUDA利用
detector = create_detector('yolov3', device='cuda:0')

# CPU利用（GPUがない環境）
detector = create_detector('yolov3', device='cpu')
```

GPUを使用すると、10倍以上の高速化が期待できます。

### バッチ処理

現在の実装は1枚ずつの処理ですが、複数画像を処理する場合はループで処理します。

```python
for image_path in image_paths:
    preds = detector(image_path)
    # 処理...
```

将来的には、バッチ推論のサポートも検討可能です。

## エラーハンドリング

### スコープエラー

mmdetとmmposeのスコープが混在すると、レジストリエラーが発生します。

```
KeyError: 'xxx is not in the xxx registry'
```

`DefaultScope.overwrite_default_scope()`で適切にスコープを管理することで回避します。

### CUDA Out of Memory

大きな画像や高解像度の処理時にメモリ不足が発生する場合があります。

対策:
- 画像をリサイズする
- バッチサイズを減らす
- GPUメモリの大きいデバイスを使用

### モデルダウンロード失敗

ネットワークエラーでモデルダウンロードが失敗した場合、手動でダウンロードできます。

```bash
cd ~/.cache/torch/hub/checkpoints/
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth
```

## 拡張性

### カスタム検出器の追加

新しい顔検出器を追加するには:

1. `configs/mmdet/`に設定ファイルを追加
2. `__init__.py`の`get_config_path()`を更新
3. `__init__.py`の`get_checkpoint_path()`を更新
4. `create_detector()`のassertを更新

### カスタムランドマークモデル

異なるランドマーク数やモデルを使用する場合:

1. `configs/mmpose/`に設定ファイルを追加
2. `dataset_info`でランドマーク数を定義
3. チェックポイントファイルを用意

### 独自の前処理・後処理

`LandmarkDetector`を継承して、カスタム処理を追加できます。

```python
class CustomDetector(LandmarkDetector):
    def __call__(self, image_or_path, boxes=None):
        # カスタム前処理
        image = self.preprocess(image_or_path)

        # 通常の検出
        preds = super().__call__(image, boxes)

        # カスタム後処理
        return self.postprocess(preds)
```

## まとめ

anime-face-detectorは、OpenMMLab 2.0エコシステムの強力な機能を活用し、モジュール化された設計により拡張性の高いアニメ顔検出ライブラリを実現しています。2段階パイプラインによる高精度な検出と、自動モデルダウンロード機能により、ユーザーは簡単に利用を開始できます。
