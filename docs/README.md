# anime-face-detector ドキュメント

anime-face-detectorのライブラリレベルの技術解説ドキュメントです。

## ドキュメント一覧

### 1. [アーキテクチャ](architecture.md)

anime-face-detectorの内部アーキテクチャと実装の詳細を解説しています。

**主な内容:**
- システム構成とディレクトリ構造
- 2段階検出パイプライン（顔検出 → ランドマーク検出）
- OpenMMLab 2.0との統合方法
- スコープ管理とAPI変更への対応
- モデルの初期化プロセス
- 推論処理フロー
- パフォーマンス最適化
- エラーハンドリング

**こんな人におすすめ:**
- 内部実装を理解したい開発者
- カスタマイズや拡張を検討している方
- OpenMMLabとの統合方法を知りたい方

---

### 2. [APIリファレンス](api_reference.md)

すべてのパブリックAPIの詳細な仕様書です。

**主な内容:**
- `create_detector()`: 検出器の作成
- `get_config_path()`: 設定ファイルパスの取得
- `get_checkpoint_path()`: チェックポイントパスの取得
- `LandmarkDetector`: メインクラスの詳細
- データ形式とランドマークインデックス
- 使用例とトラブルシューティング

**こんな人におすすめ:**
- ライブラリを使用する開発者
- API仕様を確認したい方
- パラメータの詳細を知りたい方

---

### 3. [モデルと設定ファイル](models_and_configs.md)

使用されるモデルと設定ファイルの詳細な解説です。

**主な内容:**
- モデル概要（YOLOv3, Faster R-CNN, HRNetV2）
- 各モデルのアーキテクチャ詳細
- 設定ファイルの構造と解説
- 28点ランドマークの配置
- 設定ファイルのカスタマイズ方法
- モデル選択ガイド
- パフォーマンス比較

**こんな人におすすめ:**
- モデルの詳細を知りたい方
- 設定をカスタマイズしたい開発者
- 最適なモデルを選択したい方

---

### 4. [開発ガイド](development.md)

開発環境のセットアップからコントリビューションまでのガイドです。

**主な内容:**
- 開発環境のセットアップ
- プロジェクト構造
- コーディング規約（ruff使用）
- テスト方法
- デバッグ手法
- 新機能の追加方法
- ドキュメントの更新
- リリースプロセス
- コントリビューション方法

**こんな人におすすめ:**
- プロジェクトに貢献したい開発者
- 開発環境をセットアップしたい方
- コーディング規約を確認したい方

---

## クイックスタート

### インストール

```bash
pip install openmim
mim install mmengine mmcv mmdet mmpose
pip install anime-face-detector
```

### 基本的な使用方法

```python
from anime_face_detector import create_detector
import cv2

# 検出器の作成
detector = create_detector('yolov3', device='cuda:0')

# 画像から顔とランドマークを検出
image = cv2.imread('image.jpg')
preds = detector(image)

# 結果の利用
for pred in preds:
    bbox = pred['bbox']        # [x0, y0, x1, y1, score]
    keypoints = pred['keypoints']  # [[x, y, score], ...] (28点)
    print(f"検出された顔: {bbox}")
    print(f"ランドマーク数: {len(keypoints)}")
```

詳細は [APIリファレンス](api_reference.md) を参照してください。

---

## ドキュメントの読み方

### 初めての方

1. まず [APIリファレンス](api_reference.md) で基本的な使い方を学ぶ
2. [モデルと設定ファイル](models_and_configs.md) でモデルの選択方法を確認
3. 必要に応じて [アーキテクチャ](architecture.md) で内部構造を理解

### カスタマイズしたい方

1. [アーキテクチャ](architecture.md) で全体像を把握
2. [モデルと設定ファイル](models_and_configs.md) で設定のカスタマイズ方法を確認
3. [開発ガイド](development.md) で開発環境をセットアップ

### コントリビューターの方

1. [開発ガイド](development.md) で開発環境をセットアップ
2. [アーキテクチャ](architecture.md) で内部実装を理解
3. コーディング規約に従って開発

---

## ドキュメントの構成

```
docs/
├── README.md               # このファイル（インデックス）
├── architecture.md         # アーキテクチャ解説
├── api_reference.md        # APIリファレンス
├── models_and_configs.md   # モデルと設定ファイル
└── development.md          # 開発ガイド
```

---

## 関連リンク

### 公式リソース

- [GitHub Repository](https://github.com/hysts/anime-face-detector)
- [PyPI Package](https://pypi.org/project/anime-face-detector/)
- [Colab Demo](https://colab.research.google.com/github/hysts/anime-face-detector/blob/main/demo.ipynb)
- [Hugging Face Space](https://huggingface.co/spaces/ayousanz/anime-face-detector-gpu)

### OpenMMLab

- [mmdetection](https://github.com/open-mmlab/mmdetection) - 物体検出フレームワーク
- [mmpose](https://github.com/open-mmlab/mmpose) - 姿勢推定フレームワーク
- [mmengine](https://github.com/open-mmlab/mmengine) - 基盤フレームワーク
- [mmcv](https://github.com/open-mmlab/mmcv) - コンピュータビジョンライブラリ

### コミュニティ

- [Issues](https://github.com/hysts/anime-face-detector/issues) - バグ報告、機能要望
- [Discussions](https://github.com/hysts/anime-face-detector/discussions) - 質問、アイデア共有

---

## FAQ

### Q: どのモデルを使うべきですか？

A: 用途によります。
- **リアルタイム処理**: YOLOv3（高速）
- **高精度が必要**: Faster R-CNN（高精度）

詳細は [モデルと設定ファイル](models_and_configs.md#モデルの選択ガイド) を参照してください。

### Q: GPUなしで動作しますか？

A: はい。`device='cpu'` を指定することでCPUで実行できます。

```python
detector = create_detector('yolov3', device='cpu')
```

ただし、GPUに比べて10倍以上遅くなります。

### Q: カスタムモデルを使用できますか？

A: はい。`LandmarkDetector` クラスを直接使用することで、カスタム設定とチェックポイントを指定できます。

詳細は [APIリファレンス](api_reference.md#カスタム設定での初期化) を参照してください。

### Q: ランドマークの配置は？

A: 28点のランドマークが顔の各部位に配置されます。

詳細は [モデルと設定ファイル](models_and_configs.md#ランドマーク配置詳細) を参照してください。

### Q: エラーが発生しました

A: よくあるエラーについては以下を参照してください。
- [APIリファレンス - エラーとトラブルシューティング](api_reference.md#エラーとトラブルシューティング)
- [モデルと設定ファイル - トラブルシューティング](models_and_configs.md#トラブルシューティング)
- [開発ガイド - トラブルシューティング](development.md#トラブルシューティング)

---

## ドキュメントへの貢献

ドキュメントの改善提案や誤字の修正は、Pull Requestで受け付けています。

詳細は [開発ガイド - コントリビューション](development.md#コントリビューション) を参照してください。

---

## ライセンス

このドキュメントは、anime-face-detector本体と同じMITライセンスの下で公開されています。

---

最終更新: 2026-01-20
