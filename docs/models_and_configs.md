# モデルと設定ファイル

このドキュメントでは、anime-face-detectorで使用されるモデルと設定ファイルについて詳しく解説します。

## モデル概要

anime-face-detectorは、以下の3つの事前学習済みモデルを提供しています。

| モデル | タスク | アーキテクチャ | サイズ | 用途 |
|--------|--------|----------------|--------|------|
| YOLOv3 | 顔検出 | Darknet-53 | ~120MB | 高速検出 |
| Faster R-CNN | 顔検出 | ResNet-50 + FPN | ~160MB | 高精度検出 |
| HRNetV2 | ランドマーク検出 | HRNetV2 | ~100MB | 28点ランドマーク |

## モデルのダウンロード

モデルは初回実行時に自動的にGitHub Releasesからダウンロードされます。

- **ダウンロード元**: https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1
- **保存先**: `~/.cache/torch/hub/checkpoints/`

手動ダウンロードも可能です。

```bash
cd ~/.cache/torch/hub/checkpoints/
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_faster-rcnn.pth
wget https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth
```

---

## 顔検出モデル

### YOLOv3

#### アーキテクチャ

- **バックボーン**: Darknet-53（53層の畳み込みネットワーク）
- **ネック**: YOLOV3Neck（3スケールのFPN）
- **ヘッド**: YOLOV3Head（1クラス検出）

#### 設定ファイル: configs/mmdet/yolov3.py

```python
model = dict(
    type='YOLOV3',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],  # 画像を0-1に正規化
        bgr_to_rgb=True,
        pad_size_divisor=32,        # 32の倍数にパディング
    ),
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5)       # 3つのスケールの特徴マップを出力
    ),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128],
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=1,               # アニメ顔のみ
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],  # 大きい顔用
                [(30, 61), (62, 45), (59, 119)],      # 中くらいの顔用
                [(10, 13), (16, 30), (33, 23)],       # 小さい顔用
            ],
            strides=[32, 16, 8],
        ),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
    ),
    test_cfg=dict(
        nms_pre=1000,               # NMS前に保持する最大ボックス数
        min_bbox_size=0,
        score_thr=0.05,             # スコア閾値
        conf_thr=0.005,             # 信頼度閾値
        nms=dict(type='nms', iou_threshold=0.45),  # NMS IoU閾値
        max_per_img=100,            # 画像あたり最大検出数
    ),
)
```

#### テストパイプライン

```python
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),  # アスペクト比維持でリサイズ
    dict(type='Pad', size=(608, 608), pad_val=dict(img=(114, 114, 114))),  # グレーでパディング
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
```

#### 特徴

- **速度**: 非常に高速（GPU: 30-50 FPS）
- **精度**: 良好（mAP: ~0.95）
- **メモリ使用量**: 中程度
- **推奨用途**: リアルタイムアプリケーション、動画処理

---

### Faster R-CNN

#### アーキテクチャ

- **バックボーン**: ResNet-50
- **ネック**: FPN（Feature Pyramid Network）
- **RPN**: Region Proposal Network
- **ヘッド**: Shared2FCBBoxHead（2段階検出）

#### 設定ファイル: configs/mmdet/faster-rcnn.py

主要な構成要素（YOLOv3との違い）。

```python
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,             # stage1を固定
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,           # アニメ顔のみ
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)
```

#### 特徴

- **速度**: 中速（GPU: 10-20 FPS）
- **精度**: 高精度（mAP: ~0.97）
- **メモリ使用量**: 大きめ
- **推奨用途**: 精度重視のアプリケーション、静止画処理

---

## ランドマーク検出モデル

### HRNetV2

#### アーキテクチャ

High-Resolution Network v2は、高解像度表現を維持しながらマルチスケール特徴を学習します。

```
入力 (256x256)
  ↓
Stage 1: 64チャンネル (1ブランチ)
  ↓
Stage 2: 18, 36チャンネル (2ブランチ)
  ↓
Stage 3: 18, 36, 72チャンネル (3ブランチ) ← 4回繰り返し
  ↓
Stage 4: 18, 36, 72, 144チャンネル (4ブランチ) ← 3回繰り返し
  ↓
Concat: 270チャンネル (18+36+72+144)
  ↓
Head: 28チャンネルのヒートマップ
```

#### 設定ファイル: configs/mmpose/hrnetv2.py

```python
# Codec設定（座標 ⇄ ヒートマップ変換）
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),       # 入力サイズ
    heatmap_size=(64, 64),       # ヒートマップサイズ（1/4）
    sigma=2,                     # ガウシアンカーネルの標準偏差
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet統計
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True,  # 全ブランチを出力
            ),
        ),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,                     # 全ブランチを連結
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=270,                 # 18+36+72+144
        out_channels=28,                 # 28点のランドマーク
        deconv_out_channels=None,
        conv_out_channels=(270,),
        conv_kernel_sizes=(1,),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,                 # flip_testはLandmarkDetectorで制御
    ),
)
```

#### Flip Indices（左右反転マッピング）

水平反転時に、左右対称のランドマークを入れ替えるためのインデックス。

```python
flip_indices = [
    4, 3, 2, 1, 0,        # 顔の輪郭（左右反転）
    10, 9, 8, 7, 6, 5,    # 眉（左右反転）
    19, 18, 17,           # 右目 ↔ 左目
    22, 21, 20,           # 鼻・口（左右反転）
    13, 12, 11,           # 左目 ↔ 右目
    16, 15, 14,           # 口（左右反転）
    23,                   # 中央（変わらず）
    26, 25, 24,           # 口下部（左右反転）
    27                    # 中央（変わらず）
]
```

#### Dataset Metainfo

ランドマークのメタ情報を定義します。

```python
dataset_info = dict(
    dataset_name='anime_face',
    keypoint_info={
        0: dict(name='kpt-0', id=0, color=[255, 255, 255], swap='kpt-4'),
        # ... 各ランドマークの情報
        27: dict(name='kpt-27', id=27, color=[255, 255, 255], swap=''),
    },
    skeleton_info={},           # スケルトン接続（未使用）
    joint_weights=[1.0] * 28,   # 各ランドマークの重み（すべて等価）
    sigmas=[0.025] * 28,        # OKS計算用の標準偏差
    flip_indices=flip_indices,
)
```

#### テストパイプライン

```python
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),              # ボックス中心とスケールを計算
    dict(type='TopdownAffine', input_size=(256, 256)),  # アフィン変換で正規化
    dict(type='PackPoseInputs'),
]
```

#### 特徴

- **精度**: 非常に高精度（PCK@0.1: ~0.95）
- **速度**: 中速（GPU: 50-100 FPS）
- **メモリ使用量**: 中程度
- **28点ランドマーク**: アニメ顔専用の詳細なランドマーク

---

## ランドマーク配置詳細

### 28点の配置

```
         5   6   7                 8   9  10
           ●   ●   ●           ●   ●   ●
          (左眉)                 (右眉)

    11  12  13                   17  18  19
      ●   ●   ●                   ●   ●   ●
       (左目)                      (右目)

              14  15  16
                ●   ●   ●
                 (鼻)

         20  21  22
          ●   ●   ●
           (口右)

              23
               ●
            (口中央下)

         24  25  26
          ●   ●   ●
           (口下部)

              27
               ●
           (口中央)

0   1   2                         3   4
 ●   ●   ●                         ●   ●
 (左輪郭)                          (右輪郭)
```

### インデックスと部位の対応

| インデックス | 部位 | 説明 |
|-------------|------|------|
| 0-2 | 左輪郭 | 顔の左側の輪郭 |
| 3-5 | 右輪郭 | 顔の右側の輪郭 |
| 5-7 | 左眉 | 左眉の3点 |
| 8-10 | 右眉 | 右眉の3点 |
| 11-13 | 左目 | 左目の3点 |
| 14-16 | 鼻 | 鼻の3点 |
| 17-19 | 右目 | 右目の3点 |
| 20-22 | 口右側 | 口の右側3点 |
| 23 | 口中央下 | 口の中央下部 |
| 24-26 | 口下部 | 口の下部3点 |
| 27 | 口中央 | 口の中央 |

---

## 設定ファイルのカスタマイズ

### YOLOv3のカスタマイズ例

```python
# configs/mmdet/yolov3_custom.py
_base_ = './yolov3.py'

# 入力サイズを変更
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),  # 608→416
    dict(type='Pad', size=(416, 416), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# スコア閾値を変更
model = dict(
    test_cfg=dict(
        score_thr=0.3,  # 0.05→0.3（高信頼度のみ）
    ),
)
```

### HRNetV2のカスタマイズ例

```python
# configs/mmpose/hrnetv2_custom.py
_base_ = './hrnetv2.py'

# 入力サイズを変更
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),  # 256→384（高解像度）
    heatmap_size=(96, 96),  # 64→96
    sigma=3,                # 2→3
)

# テストパイプラインを更新
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]
```

### カスタム設定の使用

```python
from anime_face_detector.detector import LandmarkDetector
from mmengine.config import Config

# カスタム設定を読み込み
custom_config = Config.fromfile('configs/mmdet/yolov3_custom.py')

detector = LandmarkDetector(
    landmark_detector_config_or_path='configs/mmpose/hrnetv2.py',
    landmark_detector_checkpoint_path='checkpoints/hrnetv2.pth',
    face_detector_config_or_path=custom_config,
    face_detector_checkpoint_path='checkpoints/yolov3.pth',
)
```

---

## モデルの選択ガイド

### 顔検出器の選択

| 要件 | 推奨モデル | 理由 |
|------|-----------|------|
| リアルタイム処理 | YOLOv3 | 高速 |
| 高精度が必要 | Faster R-CNN | 高精度 |
| 小さい顔の検出 | YOLOv3 | マルチスケール検出 |
| メモリ制約 | YOLOv3 | メモリ使用量が少ない |

### パラメータの調整

| パラメータ | 小さい値 | 大きい値 | 推奨値 |
|-----------|---------|---------|--------|
| box_scale_factor | 顔領域のみ | 広い文脈 | 1.1-1.2 |
| flip_test | False (高速) | True (高精度) | True |
| score_thr | 多くの検出 | 高信頼度のみ | 0.05-0.3 |

---

## ヒートマップの可視化

HRNetV2は、各ランドマークのヒートマップを生成します。

```python
import matplotlib.pyplot as plt
import numpy as np
from anime_face_detector import create_detector

detector = create_detector('yolov3')

# 内部のヒートマップにアクセスするには、モデルを直接使用
# （通常のAPIでは座標のみ返される）
```

実際のヒートマップは `(64, 64, 28)` の形状で、各チャンネルが1つのランドマークに対応します。

---

## パフォーマンス比較

### 顔検出

| モデル | GPU (FPS) | CPU (FPS) | mAP | サイズ |
|--------|-----------|-----------|-----|--------|
| YOLOv3 | 30-50 | 2-5 | 0.95 | 120MB |
| Faster R-CNN | 10-20 | 1-2 | 0.97 | 160MB |

### ランドマーク検出

| モデル | GPU (FPS) | CPU (FPS) | PCK@0.1 | サイズ |
|--------|-----------|-----------|---------|--------|
| HRNetV2 | 50-100 | 5-10 | 0.95 | 100MB |

GPU: NVIDIA RTX 3090、CPU: Intel Core i9-10900K での測定値（参考値）。

---

## トラブルシューティング

### モデル読み込みエラー

```
RuntimeError: Error(s) in loading state_dict
```

解決策: モデルファイルが破損している可能性があります。削除して再ダウンロードしてください。

```bash
rm ~/.cache/torch/hub/checkpoints/mmdet_anime-face_*.pth
rm ~/.cache/torch/hub/checkpoints/mmpose_anime-face_*.pth
```

### 設定ファイルエラー

```
KeyError: 'xxx is not in the config'
```

解決策: mmdet/mmposeのバージョンを確認してください。このライブラリはmmdet 3.x、mmpose 1.xに対応しています。

### CUDA設定エラー

```
RuntimeError: CUDA error: no kernel image is available
```

解決策: mmcvをCUDAアーキテクチャに合わせて再ビルドしてください。

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir
```

---

## まとめ

anime-face-detectorは、YOLOv3/Faster R-CNNによる高精度な顔検出と、HRNetV2による詳細な28点ランドマーク検出を提供します。設定ファイルをカスタマイズすることで、用途に応じた最適なパフォーマンスを実現できます。
