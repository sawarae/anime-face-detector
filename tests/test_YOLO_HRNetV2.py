import cv2
import numpy as np
import pathlib
import anime_face_detector

def test_yolo_hrnetv2():
    CUSTOM_MODEL = "YOUR_CUSTOM_MODEL.pt"
    print("=== YOLOv8とHRNetV2の統合テスト ===")
    
    # 画像を読み込む
    img = cv2.imread('tests/assets/image.png')
    assert img is not None, "画像の読み込みに失敗しました"
    print(f"画像形状: {img.shape}")
    
    # create_detectorでYOLOv8とHRNetV2を統合したモデルを作成
    # カスタムYOLOv8モデル (models/FacesV1.pt) を使用
    print("\n--- Detectorを初期化 ---")
    custom_model_path = pathlib.Path(f'models/{CUSTOM_MODEL}')
    
    if custom_model_path.exists():
        print(f"カスタムモデルを使用: {custom_model_path}")
        detector = anime_face_detector.create_detector(
            face_detector_name='yolov8',
            landmark_model_name='hrnetv2',
            device='cpu',
            custom_detector_checkpoint_path=custom_model_path,
            detector_framework='ultralytics'
        )
    else:
        print(f"カスタムモデルが見つかりません: {custom_model_path}")
        print("デフォルトのYOLOv8モデルを使用します")
        detector = anime_face_detector.create_detector(
            face_detector_name='yolov8',
            landmark_model_name='hrnetv2',
            device='cpu'
        )
    print("Detector初期化完了")
    
    # 顔検出とランドマーク推定を実行
    print("\n--- 顔検出とランドマーク推定を実行 ---")
    preds = detector(img)
    
    # 結果を表示
    print(f"検出された顔の数: {len(preds)}")
    for i, pred in enumerate(preds):
        print(f"\n顔 {i+1}:")
        print(f"  bbox: {pred['bbox']}")
        print(f"  keypoints形状: {pred['keypoints'].shape}")
        print(f"  最初のキーポイント: {pred['keypoints'][0]}")
        
        # 検出領域を保存
        box = pred['bbox']
        x1, y1, x2, y2 = np.round(box[:4]).astype(int)
        cropped_img = img[y1:y2, x1:x2]
        cv2.imwrite(f'tests/assets/detected_face_{i}.png', cropped_img)
        print(f"  検出領域を保存: tests/assets/detected_face_{i}.png")
    
    # キーポイントを描画した画像を出力
    print("\n--- キーポイントを描画 ---")
    result_img = img.copy()
    for i, pred in enumerate(preds):
        # bbox矩形を描画
        box = pred['bbox']
        x1, y1, x2, y2 = np.round(box[:4]).astype(int)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # キーポイントを描画
        keypoints = pred['keypoints']  # (K, 3) shape: [x, y, score]
        for pt in keypoints:
            x, y, score = pt
            x, y = int(x), int(y)
            # スコアに応じて色を変更
            if score > 0.5:
                color = (0, 255, 0)  # 緑: 高信頼度
            elif score > 0.3:
                color = (0, 165, 255)  # オレンジ: 中信頼度
            else:
                color = (0, 0, 255)  # 赤: 低信頼度
            cv2.circle(result_img, (x, y), 3, color, -1)
    
    cv2.imwrite('tests/assets/yolo_landmarks_output.png', result_img)
    print(f"キーポイント描画画像を保存: tests/assets/yolo_landmarks_output.png")
    
    # 結果が得られたことを確認
    assert len(preds) > 0, "顔検出に失敗しました"
    print("\n=== テスト成功 ===")

if __name__ == "__main__":
    test_yolo_hrnetv2()
