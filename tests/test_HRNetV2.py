import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from anime_face_detector import get_config_path, get_checkpoint_path

def test_hrnetv2_only():
    # HRNetV2モデルの設定とチェックポイントを取得
    config_path = get_config_path('hrnetv2')
    checkpoint_path = get_checkpoint_path('hrnetv2')

    # モデルを初期化
    model = init_model(str(config_path), str(checkpoint_path), device='cpu')

    # 適当な画像として、assets/input.jpgを使用
    img = cv2.imread('assets/input.jpg')

    # 画像が正しく読み込まれたか確認
    assert img is not None, "画像の読み込みに失敗しました"

    # 適当なbboxを設定（画像内の顔の位置を仮定）
    # 実際の使用では顔検出器からbboxを取得するが、ここではHRNetV2だけテスト
    bbox = [50, 50, 200, 200]  # x1, y1, x2, y2 の形式
    person_results = [{'bbox': bbox}]

    # HRNetV2でランドマーク推定を実行
    results = inference_topdown(model, img, person_results)

    # 結果を表示
    print(f"推定結果: {results}")
    if results:
        keypoints = results[0]['keypoints']
        print(f"キーポイントの数: {len(keypoints)}")
        print(f"キーポイントの形状: {np.array(keypoints).shape}")

    # 結果が得られたことを確認
    assert len(results) > 0, "ランドマーク推定に失敗しました"

if __name__ == "__main__":
    test_hrnetv2_only()