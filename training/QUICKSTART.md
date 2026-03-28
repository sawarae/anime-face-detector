# クイックスタートガイド

最小限の手順で動物キャラクターのランドマーク検出モデルを学習します。

## 1. セットアップ（5分）

```bash
cd training
uv pip install -e .
```

## 2. 画像収集（10分）

```bash
# Safebooruから100枚の動物キャラクター画像をダウンロード
python tools/crawl_images.py \
    --source safebooru \
    --tags "animal_ears solo rating:safe" \
    --limit 100 \
    --output-dir data/raw
```

## 3. アノテーション（人間が作業）

```bash
# アノテーションツールを起動
python tools/annotate_landmarks.py \
    --image-dir data/raw \
    --output-dir data/annotated
```

1. ブラウザで http://localhost:7860 を開く
2. 画像をクリックして28点のランドマークをマーク
3. 「Save Annotation」をクリック
4. 「Next」で次の画像へ

**所要時間**: 1画像あたり1-2分（100枚で2-3時間）

## 4. 学習（30分〜数時間）

```bash
# 既存モデルから転移学習（推奨）
python scripts/train.py \
    --image-dir data/raw \
    --annotation-dir data/annotated \
    --checkpoint ../anime_face_detector/models/weights/mmpose_anime-face_hrnetv2.pth \
    --output-dir checkpoints/animal_v1 \
    --epochs 50 \
    --batch-size 8 \
    --device cuda
```

**Note**: 既存モデルのパスは環境に応じて調整してください。

## 5. 評価（5分）

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/animal_v1/best_model.pth \
    --image-dir data/raw \
    --annotation-dir data/annotated \
    --visualize
```

結果は `eval_results/` に保存されます。

## 6. モデルを本体で使用

```bash
# state_dictを抽出
python -c "
import torch
checkpoint = torch.load('checkpoints/animal_v1/best_model.pth', weights_only=True)
torch.save(checkpoint['state_dict'], 'animal_hrnetv2.pth')
"
```

```python
# Pythonコードで使用
from anime_face_detector import create_detector

detector = create_detector(
    landmark_model_checkpoint_path='training/animal_hrnetv2.pth',
    device='cuda:0'
)

# 推論
import cv2
image = cv2.imread('test_image.jpg')
results = detector(image)
```

## よくある質問

**Q: どのくらいのデータが必要？**
A: 最低100枚、推奨500枚以上。データが多いほど精度が向上します。

**Q: GPUは必須？**
A: 必須ではありませんが、GPUがあると学習が大幅に高速化されます（CPUの10-50倍）。

**Q: アノテーションを外注できる？**
A: はい。JSON形式で保存されるため、複数人で分担可能です。

**Q: 既存モデルがない場合は？**
A: `--checkpoint`オプションを省略してスクラッチから学習できますが、収束に時間がかかります。

**Q: 学習が収束しない**
A: 学習率を下げる（`--lr 0.00001`）、データを増やす、転移学習を使用する、などを試してください。

## 次のステップ

- より多くのデータを収集して精度向上
- データ拡張の実装
- ハイパーパラメータのチューニング
- 複数の動物種に対応したモデルの学習

詳細は [README.md](README.md) を参照してください。
