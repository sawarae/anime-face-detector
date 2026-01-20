import cv2
import numpy as np
import anime_face_detector

def test_yolo_hrnetv2():
    print("=== YOLOv8とHRNetV2の統合テスト ===")
    
    # 画像を読み込む
    img = cv2.imread('tests/assets/image.png')
    assert img is not None, "画像の読み込みに失敗しました"
    print(f"画像形状: {img.shape}")
    
    # create_detectorでYOLOv8とHRNetV2を統合したモデルを作成
    print("\n--- Detectorを初期化 ---")
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
    
    # 結果が得られたことを確認
    assert len(preds) > 0, "顔検出に失敗しました"
    print("\n=== テスト成功 ===")

if __name__ == "__main__":
    test_yolo_hrnetv2()
