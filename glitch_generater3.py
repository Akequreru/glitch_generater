import cv2
import mediapipe as mp
import numpy as np
import os
import random

# --- 設定 ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Face Meshのランドマークインデックス
LM_INDICES = {
    'top': 10, 'bottom': 152, 'left_face': 234, 'right_face': 454,
    # 左目
    'left_eye_top': [33, 246, 161, 160, 159, 158, 157, 173, 133],
    'left_eye_bottom': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    # 右目
    'right_eye_top': [362, 398, 384, 385, 386, 387, 388, 466, 263],
    'right_eye_bottom': [362, 382, 381, 380, 374, 373, 390, 249, 263]
}

# --- 関数群 ---
def apply_eyelid_movement(roi, top_pts, bottom_pts, power=0.6, sigma_scale=2.0):
    """
    まぶたの皮膚を中心から外側へ物理的に押し広げる処理
    power: 引っ張る強さ
    sigma_scale: 引っ張る範囲の広さ（値が小さいと鋭く、大きいと広範囲が動く）
    """
    h, w = roi.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)

    all_pts = np.array(top_pts + bottom_pts)
    if len(all_pts) == 0: return roi.copy()
    
    center_y = np.mean(all_pts[:, 1])
    eye_h = np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])
    if eye_h <= 0: eye_h = 1.0

    # 中心からの距離
    delta_y = map_y - center_y
    
    # sigma_scaleを使って影響範囲をコントロール
    sigma_y = eye_h * sigma_scale 
    
    # ガウシアン重み
    weight = np.exp(-(delta_y**2 / (2 * sigma_y**2)))

    # 逆写像計算
    scaling = 1.0 - (power * weight)
    map_y_new = center_y + delta_y * scaling

    # リマップ適用
    moved_roi = cv2.remap(roi, map_x, map_y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return moved_roi

def create_single_eye_mask(h, w, contour_pts, roi_offset):
    """ROIサイズの片目マスクを作成(白黒)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    off_x, off_y = roi_offset
    pts_rel = [(pt[0] - off_x, pt[1] - off_y) for pt in contour_pts]
    if not pts_rel: return mask
    cv2.fillPoly(mask, [np.array(pts_rel, dtype=np.int32)], 255)
    return mask

def get_sclera_color(roi, mask):
    """白目の色をスポイト"""
    valid_pixels_mask = mask > 220
    if not np.any(valid_pixels_mask): return (200, 200, 200)
    pixels = roi[valid_pixels_mask]
    brightness = pixels[:, 0] * 0.114 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.299
    sorted_pixels = pixels[np.argsort(brightness)]
    num = len(sorted_pixels)
    start, end = int(num * 0.50), int(num * 0.90)
    if start >= end: start, end = int(num * 0.4), int(num * 0.6)
    selected = sorted_pixels[start:end]
    if len(selected) == 0: return (200, 200, 200)
    return tuple(map(int, np.mean(selected, axis=0)))

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    """YOLOフォーマット保存"""
    xmin, ymin, xmax, ymax = bbox_coords
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0: return
    xc, yc = xmin + bw / 2.0, ymin + bh / 2.0
    with open(filepath, "w") as f:
        f.write(f"0 {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

# --- メイン処理 ---
def process_images():
    mp_face_mesh = mp.solutions.face_mesh
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.5
    ) as face_mesh:
        
        for filename in files:
            input_path = os.path.join(INPUT_DIR, filename)
            file_bytes = np.fromfile(input_path, np.uint8)
            img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_original is None: continue
            
            print(f"--- 処理開始: {filename} ---")
            h_img, w_img, _ = img_original.shape
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks: continue

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # 座標取得
                pts = {}
                for key, indices in LM_INDICES.items():
                    if isinstance(indices, int):
                        lm = face_landmarks.landmark[indices]
                        pts[key] = (int(lm.x * w_img), int(lm.y * h_img))
                    else:
                        pts[key] = [(int(face_landmarks.landmark[idx].x * w_img), int(face_landmarks.landmark[idx].y * h_img)) for idx in indices]

                # ROI決定
                face_coords = [pts['top'], pts['bottom'], pts['left_face'], pts['right_face']]
                xs, ys = [p[0] for p in face_coords], [p[1] for p in face_coords]
                face_xmin, face_ymin, face_xmax, face_ymax = min(xs), min(ys), max(xs), max(ys)
                face_w, face_h = face_xmax - face_xmin, face_ymax - face_ymin
                
                margin = 0.4
                roi_cx, roi_cy = (face_xmin + face_xmax) // 2, (face_ymin + face_ymax) // 2
                roi_size = max(face_w, face_h) * (1 + margin)
                x1, y1 = max(0, int(roi_cx - roi_size/2)), max(0, int(roi_cy - roi_size/2))
                x2, y2 = min(w_img, int(roi_cx + roi_size/2)), min(h_img, int(roi_cy + roi_size/2))
                
                roi_src = img_original[y1:y2, x1:x2]
                if roi_src.shape[0] == 0 or roi_src.shape[1] == 0: continue
                roi_h, roi_w, _ = roi_src.shape
                roi_offset = (x1, y1)
                
                pts_local_top_l = [(p[0]-x1, p[1]-y1) for p in pts['left_eye_top']]
                pts_local_bottom_l = [(p[0]-x1, p[1]-y1) for p in pts['left_eye_bottom']]
                pts_local_top_r = [(p[0]-x1, p[1]-y1) for p in pts['right_eye_top']]
                pts_local_bottom_r = [(p[0]-x1, p[1]-y1) for p in pts['right_eye_bottom']]

                # --- ランダムパラメータ設定 (ここを変更) ---
                # 強さ(Power): 0.3(微弱) 〜 1.1(崩壊レベル)
                base_power = random.uniform(0.3, 1.1)
                # 範囲(Sigma): 1.5(鋭い) 〜 3.0(広範囲)
                base_sigma = random.uniform(1.5, 3.0)

                # 左右非対称性 (Asymmetry): ±10%程度のばらつきを与える
                power_l = base_power * random.uniform(0.3, 1.1)
                power_r = base_power * random.uniform(0.3, 1.1)
                sigma_l = base_sigma * random.uniform(0.9, 1.1)
                sigma_r = base_sigma * random.uniform(0.9, 1.1)

                name, ext = os.path.splitext(filename)
                bbox_coords = (face_xmin, face_ymin, face_xmax, face_ymax)
                # ファイル名には平均的なパワーを記載
                power_str = f"p{int(base_power*100)}"

                # ==================================================
                # 準備: マスク作成と色スポイト
                # ==================================================
                orig_contour_l = pts['left_eye_top'] + pts['left_eye_bottom'][::-1]
                orig_contour_r = pts['right_eye_top'] + pts['right_eye_bottom'][::-1]
                mask_l_roi = create_single_eye_mask(roi_h, roi_w, orig_contour_l, roi_offset)
                mask_r_roi = create_single_eye_mask(roi_h, roi_w, orig_contour_r, roi_offset)
                
                color_l = get_sclera_color(roi_src, mask_l_roi)
                color_r = get_sclera_color(roi_src, mask_r_roi)

                # ==================================================
                # ルート1: 白目化ベース
                # ==================================================
                # 1-1. 白目塗りつぶし (変形なし)
                blur_size = (5, 5)
                mask_l_3ch = cv2.cvtColor(cv2.GaussianBlur(mask_l_roi, blur_size, 0), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
                mask_r_3ch = cv2.cvtColor(cv2.GaussianBlur(mask_r_roi, blur_size, 0), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
                fill_l = np.full_like(roi_src, color_l).astype(np.float32)
                fill_r = np.full_like(roi_src, color_r).astype(np.float32)
                base = roi_src.astype(np.float32)
                blank_eyes_roi = np.clip(base*(1-mask_l_3ch) + fill_l*mask_l_3ch, 0, 255).astype(np.float32)
                blank_eyes_roi = np.clip(blank_eyes_roi*(1-mask_r_3ch) + fill_r*mask_r_3ch, 0, 255).astype(np.uint8)

                # 保存: 白目のみ
                output_img_blank_only = img_original.copy()
                output_img_blank_only[y1:y2, x1:x2] = blank_eyes_roi
                base_name_blank_only = f"{name}_blank_only_{i}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name_blank_only}{ext}"), output_img_blank_only)
                save_yolo_format(os.path.join(OUTPUT_DIR, f"{base_name_blank_only}.txt"), w_img, h_img, bbox_coords)
                print(f"    -> 保存完了(白目のみ): {base_name_blank_only}")

                # 1-2. 白目画像に対して見開き変形 (ランダムパラメータ適用)
                # 左目変形
                temp_roi = apply_eyelid_movement(
                    blank_eyes_roi, pts_local_top_l, pts_local_bottom_l, power=power_l, sigma_scale=sigma_l
                )
                # 右目変形 (temp_roiを入力)
                final_roi_blank_open = apply_eyelid_movement(
                    temp_roi, pts_local_top_r, pts_local_bottom_r, power=power_r, sigma_scale=sigma_r
                )

                # 保存: 白目＋見開き
                output_img_blank_open = img_original.copy()
                output_img_blank_open[y1:y2, x1:x2] = final_roi_blank_open
                base_name_blank_open = f"{name}_blank_open_{power_str}_{i}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name_blank_open}{ext}"), output_img_blank_open)
                save_yolo_format(os.path.join(OUTPUT_DIR, f"{base_name_blank_open}.txt"), w_img, h_img, bbox_coords)
                print(f"    -> 保存完了(白目見開き): {base_name_blank_open} (Pow:{base_power:.2f}, Sig:{base_sigma:.2f})")

                # ==================================================
                # ルート2: 元の目ベース
                # ==================================================
                # 元の画像に対して見開き変形 (ランダムパラメータ適用)
                temp_roi_orig = apply_eyelid_movement(
                    roi_src, pts_local_top_l, pts_local_bottom_l, power=power_l, sigma_scale=sigma_l
                )
                final_roi_original_open = apply_eyelid_movement(
                    temp_roi_orig, pts_local_top_r, pts_local_bottom_r, power=power_r, sigma_scale=sigma_r
                )

                # 保存: 元目＋見開き
                output_img_orig_open = img_original.copy()
                output_img_orig_open[y1:y2, x1:x2] = final_roi_original_open
                base_name_orig_open = f"{name}_original_open_{power_str}_{i}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name_orig_open}{ext}"), output_img_orig_open)
                save_yolo_format(os.path.join(OUTPUT_DIR, f"{base_name_orig_open}.txt"), w_img, h_img, bbox_coords)
                print(f"    -> 保存完了(元目見開き): {base_name_orig_open}")

    print("\nすべての処理が完了しました。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()