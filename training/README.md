# HRNetV2 Training Tools for Animal Character Landmark Detection

このディレクトリには、動物キャラクターの特徴点検出のためのHRNetV2モデルを学習するためのツールが含まれています。

## ディレクトリ構成

```
training/
├── tools/              # ツール
│   ├── crawl_images.py        # 画像クローリング
│   ├── annotate_landmarks.py  # アノテーションツール（Gradio）
│   └── dataset.py             # PyTorchデータセット
├── scripts/            # 学習・評価スクリプト
│   ├── train.py               # 学習スクリプト
│   └── evaluate.py            # 評価スクリプト
├── configs/            # 設定ファイル
│   └── default.yaml           # デフォルト設定
├── data/               # データディレクトリ
│   ├── raw/                   # ダウンロードした画像
│   ├── annotated/             # アノテーション済みJSON
│   └── datasets/              # train/val/test分割
└── checkpoints/        # 学習済みモデル
```

## セットアップ

### 1. 環境構築

```bash
cd training

# uv で依存関係をインストール
uv pip install -e .

# または pip を使用
pip install -e .
```

### 2. 必要な依存関係

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Gradio >= 5.0.0
- gallery-dl >= 1.26.0
- その他（pyproject.toml参照）

## ワークフロー

### ステップ1: 画像のクローリング

Safebooru、Danbooru、Gelbooruから画像を収集します。

```bash
# Safebooruから動物キャラクター画像を100枚ダウンロード
python tools/crawl_images.py \
    --source safebooru \
    --tags "animal_ears solo 1girl" \
    --limit 100 \
    --output-dir data/raw

# Furry系のタグで検索
python tools/crawl_images.py \
    --source safebooru \
    --tags "furry anthro" \
    --limit 200 \
    --output-dir data/raw/furry

# Gelbooruを使用
python tools/crawl_images.py \
    --source gelbooru \
    --tags "kemono_friends" \
    --limit 150 \
    --output-dir data/raw
```

**注意**: `gallery-dl`が必要です。事前にインストールしてください。

```bash
pip install gallery-dl
```

### ステップ2: アノテーション

Gradioベースのアノテーションツールで28点のランドマークをマークします。

```bash
# アノテーションツールを起動
python tools/annotate_landmarks.py \
    --image-dir data/raw \
    --output-dir data/annotated \
    --port 7860
```

ブラウザで http://localhost:7860 を開き、アノテーションを開始します。

#### アノテーション手順

1. 画像をクリックして、順番にランドマークを追加（0-27）
2. 28点すべてをマークしたら「Save Annotation」ボタンをクリック
3. 「Next」ボタンで次の画像へ
4. 間違えた場合は「Clear Last」または「Clear All」で修正

#### ランドマークの順序（28点）

- **0-4**: 顔輪郭（青）
- **5-10**: 眉（緑）
- **11-13, 17-19**: 目（黄）
- **14-16**: 鼻（マゼンタ）
- **20-27**: 口（赤）

アノテーションは`data/annotated/`にJSON形式で保存されます。

### ステップ3: データセットの準備

アノテーション済みデータをtrain/val/testに分割します。

```bash
python tools/dataset.py \
    --annotation-dir data/annotated \
    --output-dir data/datasets \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1
```

これにより、`data/datasets/`に`train.txt`, `val.txt`, `test.txt`が生成されます。

### ステップ4: 学習

HRNetV2モデルを学習します。既存の人間キャラクター用モデルから転移学習することを推奨します。

```bash
# 既存モデルから転移学習
python scripts/train.py \
    --image-dir data/raw \
    --annotation-dir data/annotated \
    --checkpoint ../anime_face_detector/models/weights/mmpose_anime-face_hrnetv2.pth \
    --output-dir checkpoints/animal_chars \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.0001 \
    --device cuda

# スクラッチから学習（非推奨）
python scripts/train.py \
    --image-dir data/raw \
    --annotation-dir data/annotated \
    --output-dir checkpoints/from_scratch \
    --epochs 200 \
    --batch-size 8 \
    --lr 0.001
```

#### 学習のオプション

- `--checkpoint`: 転移学習用の事前学習済みモデル
- `--val-annotation-dir`: 別の検証セットを使用する場合
- `--epochs`: 学習エポック数
- `--batch-size`: バッチサイズ（GPU メモリに応じて調整）
- `--lr`: 学習率
- `--save-freq`: チェックポイント保存頻度

#### TensorBoardでモニタリング

```bash
tensorboard --logdir checkpoints/animal_chars/runs
```

### ステップ5: 評価

学習済みモデルを評価します。

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/animal_chars/best_model.pth \
    --image-dir data/raw \
    --annotation-dir data/annotated \
    --output-dir eval_results \
    --visualize \
    --num-vis-samples 20
```

評価メトリクス:
- **NME (Normalized Mean Error)**: バウンディングボックスまたは瞳孔間距離で正規化された誤差

結果は`eval_results/metrics.txt`に保存され、可視化画像は`eval_results/visualizations/`に保存されます。

### ステップ6: モデルのエクスポート

学習済みモデルを本体ライブラリで使用できる形式にエクスポートします。

```bash
# チェックポイントから state_dict のみを抽出
python -c "
import torch
checkpoint = torch.load('checkpoints/animal_chars/best_model.pth', weights_only=True)
torch.save(checkpoint['state_dict'], 'animal_character_hrnetv2.pth')
"
```

このモデルを本体ライブラリで使用:

```python
from anime_face_detector import create_detector

detector = create_detector(
    landmark_model_checkpoint_path='training/animal_character_hrnetv2.pth',
    device='cuda:0'
)
```

## ヒントとベストプラクティス

### データ収集

1. **多様性を確保**: 異なる角度、表情、照明条件の画像を集める
2. **品質重視**: 解像度が低い画像や大きく遮蔽されている画像は避ける
3. **バランス**: 異なる動物種のキャラクターを均等に集める

推奨データセットサイズ:
- 最低: 100-200画像
- 推奨: 500-1000画像
- 理想: 2000画像以上

### アノテーション

1. **一貫性**: ランドマークの位置を一貫して配置
2. **精度**: 拡大表示を使用して正確にマーク
3. **休憩**: 集中力を保つため、定期的に休憩を取る

### 学習

1. **転移学習を使用**: 既存の人間キャラクターモデルから開始すると収束が早い
2. **データ拡張**: 将来的に実装予定（回転、反転、色調変換など）
3. **学習率調整**: 転移学習の場合は小さい学習率（1e-4）を使用
4. **Early Stopping**: 検証損失が改善しなくなったら学習を停止

## トラブルシューティング

### GPU メモリ不足

```bash
# バッチサイズを減らす
python scripts/train.py --batch-size 4 ...

# または勾配累積を使用（今後実装予定）
```

### 学習が収束しない

1. 学習率を下げる（1e-4 → 1e-5）
2. より多くのデータを収集
3. データ拡張を追加
4. 転移学習を使用

### アノテーションツールが起動しない

```bash
# Gradioのバージョンを確認
pip install "gradio>=5.0.0" --upgrade

# ポートを変更
python tools/annotate_landmarks.py --port 8080
```

## 参考資料

- [HRNet論文](https://arxiv.org/abs/1902.09212)
- [mmpose](https://github.com/open-mmlab/mmpose)
- [gallery-dl](https://github.com/mikf/gallery-dl)

## ライセンス

本体ライブラリと同じライセンスに従います。

## 貢献

バグ報告や改善提案は Issue または Pull Request でお願いします。
