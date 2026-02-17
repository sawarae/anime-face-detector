import cv2
import numpy as np
import torch

from anime_face_detector import get_checkpoint_path
from anime_face_detector.models import HRNetV2


def test_hrnetv2_only():
    print("=== HRNetV2単独テスト (pure PyTorch) ===")

    # 画像を読み込む
    img = cv2.imread("tests/assets/image.png")
    assert img is not None, "画像の読み込みに失敗しました"
    print(f"画像形状: {img.shape}")

    # HRNetV2チェックポイントを取得（mmpose/mmcv不要）
    checkpoint_path = get_checkpoint_path("hrnetv2")

    # モデルを初期化
    print("\n--- HRNetV2モデルを初期化 ---")
    device = "cpu"
    model = HRNetV2(num_keypoints=28, pretrained=str(checkpoint_path)).to(device).eval()
    print("HRNetV2初期化完了")

    # 適当なbboxを設定（画像内の顔の位置を仮定）
    # (N, 4) 形式: x1, y1, x2, y2
    bbox_coords = np.array([[0, 0, 800, 800]], dtype=np.float32)

    # bboxで切り取った領域を保存（確認用）
    x1, y1, x2, y2 = bbox_coords[0].astype(int)
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    assert x2 > x1 and y2 > y1, "bboxが不正です"

    cropped_img = img[y1:y2, x1:x2]
    cv2.imwrite("tests/assets/cropped_face.png", cropped_img)
    print("切り取り領域を保存: tests/assets/cropped_face.png")

    # HRNetV2でランドマーク推定を実行
    print("\n--- ランドマーク推定を実行 (pure PyTorch) ---")

    # HRNetV2は 256x256 入力を想定（実装に合わせる）
    input_size = 256
    resized = cv2.resize(cropped_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # BGR uint8 [0..255] -> float tensor (1,3,H,W)
    x = torch.from_numpy(resized).to(torch.float32)  # (H,W,3)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        x = model.preprocess(x)
        heatmaps = model(x)  # (1, K, 64, 64)
        kpts_hm, scores = model.decode_heatmaps(heatmaps)  # (1,K,2), (1,K)

    kpts_hm = kpts_hm[0].cpu().numpy()  # (K,2) in heatmap coords (x,y) on 64x64
    scores = scores[0].cpu().numpy()  # (K,)

    # heatmap(64x64) -> input(256x256): *4
    kpts_in = kpts_hm * (input_size / heatmaps.shape[-1])

    # input(256x256) -> crop(original bbox size)
    crop_h, crop_w = cropped_img.shape[:2]
    scale_x = crop_w / input_size
    scale_y = crop_h / input_size
    kpts_crop = np.stack([kpts_in[:, 0] * scale_x, kpts_in[:, 1] * scale_y], axis=1)

    # crop -> original image coordinates
    kpts_img = kpts_crop + np.array([x1, y1], dtype=np.float32)

    keypoints_with_scores = np.concatenate([kpts_img, scores[:, None]], axis=1)  # (K,3)

    preds = [{"bbox": bbox_coords[0], "keypoints": keypoints_with_scores}]

    # 結果を表示
    print(f"推定結果数: {len(preds)}")
    keypoints = preds[0]["keypoints"]
    print(f"キーポイント数: {len(keypoints)}")
    print(f"キーポイント形状: {keypoints.shape}")
    print(f"最初のキーポイント: {keypoints[0]}")

    # キーポイントを描画した画像を出力
    print("\n--- キーポイントを描画 ---")
    result_img = img.copy()
    for pt in keypoints:
        xk, yk, score = pt
        xk, yk = int(xk), int(yk)
        if score > 0.5:
            color = (0, 255, 0)
        elif score > 0.3:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(result_img, (xk, yk), 3, color, -1)

    cv2.imwrite("tests/assets/landmarks_output.png", result_img)
    print("キーポイント描画画像を保存: tests/assets/landmarks_output.png")

    assert len(preds) > 0, "ランドマーク推定に失敗しました"
    print("\n=== テスト成功 ===")


if __name__ == "__main__":
    test_hrnetv2_only()
