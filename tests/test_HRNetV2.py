import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from mmengine.registry import DefaultScope
from anime_face_detector import get_config_path, get_checkpoint_path

def test_hrnetv2_only():
    print("=== HRNetV2単独テスト ===")
    
    # 画像を読み込む
    img = cv2.imread('tests/assets/image.png')
    assert img is not None, "画像の読み込みに失敗しました"
    print(f"画像形状: {img.shape}")
    
    # HRNetV2モデルの設定とチェックポイントを取得
    config_path = get_config_path('hrnetv2')
    checkpoint_path = get_checkpoint_path('hrnetv2')

    # モデルを初期化
    print("\n--- HRNetV2モデルを初期化 ---")
    model = init_model(str(config_path), str(checkpoint_path), device='cpu')
    print("HRNetV2初期化完了")

    # 適当なbboxを設定（画像内の顔の位置を仮定）
    bbox_coords = np.array([[0, 0, 800, 800]])  # (N, 4) 形式: x1, y1, x2, y2

    # 確認用にbboxで切り取った領域を保存
    x1, y1, x2, y2 = bbox_coords[0]
    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite('tests/assets/cropped_face.png', cropped_img)
    print(f"切り取り領域を保存: tests/assets/cropped_face.png")

    # HRNetV2でランドマーク推定を実行
    print("\n--- ランドマーク推定を実行 ---")
    with DefaultScope.overwrite_default_scope('mmpose'):
        results = inference_topdown(
            model, img, bbox_coords[:, :4], bbox_format='xyxy'
        )
    
    # Convert PoseDataSample to dict format for backward compatibility
    preds = []
    for i, result in enumerate(results):
        pred_instances = result.pred_instances
        keypoints = pred_instances.keypoints[0]  # (K, 2)
        keypoint_scores = pred_instances.keypoint_scores[0]  # (K,)
        # Combine keypoints and scores to [x, y, score] format
        keypoints_with_scores = np.concatenate(
            [keypoints, keypoint_scores[:, np.newaxis]], axis=1
        )
        preds.append({'bbox': bbox_coords[i], 'keypoints': keypoints_with_scores})

    # 結果を表示
    print(f"推定結果数: {len(preds)}")
    if preds:
        keypoints = preds[0]['keypoints']
        print(f"キーポイント数: {len(keypoints)}")
        print(f"キーポイント形状: {keypoints.shape}")
        print(f"最初のキーポイント: {keypoints[0]}")

    # 結果が得られたことを確認
    assert len(preds) > 0, "ランドマーク推定に失敗しました"
    print("\n=== テスト成功 ===")

if __name__ == "__main__":
    test_hrnetv2_only()