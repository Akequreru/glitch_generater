import cv2
import mediapipe as mp
import numpy as np
import os
import random

# --- 設定 ---
INPUT_DIR = "input"
TEXTURE_DIR = "texture_input" # テクスチャ画像のフォルダ
OUTPUT_DIR = "output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TEXTURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 調整用パラメータ (前回の設定を維持)
COLOR_THRESHOLDS = {
    'h_tol': 20,
    's_tol_lower': 50,
    's_tol_upper': 80,
    'v_tol_lower': 120,
    'v_tol_upper': 100, 
    'cr_tol': 20, 
    'cb_tol': 20,
    # 目より上で髪のハイライトを弾くための彩度下限
    'min_saturation_strict': 15,
    # ハイライト（テカリ）とみなす明るさの閾値
    'highlight_threshold': 90 
}

# MediaPipe Face Mesh Indices
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356]
SKIN_SAMPLE_INDICES = [4, 234, 454, 152, 10]
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIPS_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

def get_landmark_points(landmarks, indices, w, h):
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append((int(lm.x * w), int(lm.y * h)))
    return points

def extend_forehead_points(points, indices_to_extend, face_h, extension_ratio=0.15):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cy = sum(ys) / len(ys)
    min_y = min(ys)
    shift_amount = int(face_h * extension_ratio)
    modified_points = []
    for (x, y) in points:
        if y < cy and (y - min_y) < (face_h * 0.3):
            new_y = max(0, y - shift_amount)
            modified_points.append((x, new_y))
        else:
            modified_points.append((x, y))
    return modified_points

def create_geometric_mask(roi_shape, face_pts, left_eye_pts, right_eye_pts, lips_pts, left_brow_pts, right_brow_pts, offset):
    h, w = roi_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    off_x, off_y = offset
    def to_roi(pts): return np.array([(p[0]-off_x, p[1]-off_y) for p in pts], dtype=np.int32)

    cv2.fillPoly(mask, [to_roi(face_pts)], 255)
    dilate_size = int(max(w, h) * 0.02)
    if dilate_size < 3: dilate_size = 3
    mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)

    cv2.fillPoly(mask, [to_roi(left_eye_pts)], 0)
    cv2.fillPoly(mask, [to_roi(right_eye_pts)], 0)
    cv2.fillPoly(mask, [to_roi(lips_pts)], 0)

    brow_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(brow_mask, [to_roi(left_brow_pts)], 255)
    cv2.fillPoly(brow_mask, [to_roi(right_brow_pts)], 255)
    erode_size = int(max(w, h) * 0.01) 
    if erode_size < 1: erode_size = 1
    brow_mask = cv2.erode(brow_mask, np.ones((erode_size, erode_size), np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(brow_mask))

    return mask

def create_advanced_skin_mask_with_position(roi, face_landmarks, w_img, h_img, offset, eye_level_y_rel):
    h_roi, w_roi = roi.shape[:2]
    off_x, off_y = offset
    cfg = COLOR_THRESHOLDS

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    
    sampled_hsvs = []
    sampled_ycrcbs = []
    for idx in SKIN_SAMPLE_INDICES:
        lm = face_landmarks.landmark[idx]
        lx, ly = int(lm.x * w_img) - off_x, int(lm.y * h_img) - off_y
        if 0 <= lx < w_roi and 0 <= ly < h_roi:
            y_min, y_max = max(0, ly-2), min(h_roi, ly+3)
            x_min, x_max = max(0, lx-2), min(w_roi, lx+3)
            if y_max > y_min and x_max > x_min:
                sampled_hsvs.append(np.mean(hsv_roi[y_min:y_max, x_min:x_max], axis=(0, 1)))
                sampled_ycrcbs.append(np.mean(ycrcb_roi[y_min:y_max, x_min:x_max], axis=(0, 1)))
    
    if not sampled_hsvs: return np.ones((h_roi, w_roi), dtype=np.uint8) * 255

    mean_hsv = np.mean(sampled_hsvs, axis=0)
    mean_ycrcb = np.mean(sampled_ycrcbs, axis=0)
    
    lower_hsv = np.array([max(0, mean_hsv[0]-cfg['h_tol']), max(0, mean_hsv[1]-cfg['s_tol_lower']), max(0, mean_hsv[2]-cfg['v_tol_lower'])])
    upper_hsv = np.array([min(180, mean_hsv[0]+cfg['h_tol']), min(255, mean_hsv[1]+cfg['s_tol_upper']), min(255, mean_hsv[2]+cfg['v_tol_upper'])])
    mask_hsv_base = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
    
    lower_ycrcb = np.array([0, max(0, mean_ycrcb[1]-cfg['cr_tol']), max(0, mean_ycrcb[2]-cfg['cb_tol'])])
    upper_ycrcb = np.array([255, min(255, mean_ycrcb[1]+cfg['cr_tol']), min(255, mean_ycrcb[2]+cfg['cb_tol'])])
    mask_ycrcb_base = cv2.inRange(ycrcb_roi, lower_ycrcb, upper_ycrcb)
    
    base_mask = cv2.bitwise_and(mask_hsv_base, mask_ycrcb_base)
    
    v_channel = hsv_roi[:, :, 2]
    _, mask_highlight = cv2.threshold(v_channel, cfg['highlight_threshold'], 255, cv2.THRESH_BINARY)

    grid_y, _ = np.meshgrid(np.arange(h_roi), np.arange(w_roi), indexing='ij')
    mask_area_upper = (grid_y < eye_level_y_rel).astype(np.uint8) * 255
    mask_area_lower = (grid_y >= eye_level_y_rel).astype(np.uint8) * 255
    
    s_channel = hsv_roi[:, :, 1]
    mask_sat_strict = cv2.inRange(s_channel, cfg['min_saturation_strict'], 255)
    
    upper_standard = cv2.bitwise_and(base_mask, mask_sat_strict)
    upper_highlight_recovery = cv2.bitwise_and(mask_highlight, mask_ycrcb_base)
    upper_final = cv2.bitwise_or(upper_standard, upper_highlight_recovery)
    upper_result = cv2.bitwise_and(upper_final, mask_area_upper)
    
    lower_final = cv2.bitwise_or(base_mask, mask_highlight)
    lower_result = cv2.bitwise_and(lower_final, mask_area_lower)
    
    final_mask = cv2.bitwise_or(upper_result, lower_result)
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return final_mask

def create_hybrid_mask(geo_mask, skin_mask, eye_level_y):
    h, w = geo_mask.shape
    grid_y, _ = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    mask_upper_area = (grid_y < eye_level_y).astype(np.uint8) * 255
    mask_lower_area = (grid_y >= eye_level_y).astype(np.uint8) * 255
    
    upper_result = cv2.bitwise_and(geo_mask, skin_mask)
    upper_result = cv2.bitwise_and(upper_result, mask_upper_area)
    lower_result = cv2.bitwise_and(geo_mask, mask_lower_area)
    
    final_mask = cv2.bitwise_or(upper_result, lower_result)
    final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
    return final_mask

# ★★★ テクスチャ適用関数 ★★★
def apply_texture_pattern(roi, mask, texture_img):
    """
    肌領域にテクスチャ画像の明度を合成する
    """
    h, w = roi.shape[:2]
    th, tw = texture_img.shape[:2]
    
    # テクスチャがROIより小さい場合は拡大する
    if th < h or tw < w:
        scale = max(h / th, w / tw) * 1.5 # 余裕を持って拡大
        texture_img = cv2.resize(texture_img, None, fx=scale, fy=scale)
        th, tw = texture_img.shape[:2]
        
    # ランダムな位置を切り出し
    x_start = random.randint(0, tw - w)
    y_start = random.randint(0, th - h)
    crop = texture_img[y_start:y_start+h, x_start:x_start+w]
    
    # 1. テクスチャをグレースケール化（明度抽出）
    gray_tex = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 2. 合成用データ作成
    # テクスチャの平均輝度からの偏差を計算 (明るい部分はプラス、暗い部分はマイナス)
    tex_mean = np.mean(gray_tex)
    tex_diff = gray_tex - tex_mean
    
    # コントラスト（模様の強さ）をランダム調整
    # 0.5(控えめ) 〜 1.5(くっきり)
    strength = random.uniform(0.5, 1.5)
    tex_diff = tex_diff * strength
    
    # 3. 元画像の明度(V)に加算
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv_roi)
    
    v_new = v_ch + tex_diff
    v_new = np.clip(v_new, 0, 255)
    
    # 4. 再結合
    hsv_new = cv2.merge([h_ch, s_ch, v_new])
    textured_roi = cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 5. マスク合成
    mask_3ch = cv2.merge([mask, mask, mask])
    roi_float = roi.astype(np.float32)
    tex_float = textured_roi.astype(np.float32)
    
    output = tex_float * mask_3ch + roi_float * (1.0 - mask_3ch)
    
    return output.astype(np.uint8)

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    xmin, ymin, xmax, ymax = bbox_coords
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(img_w, xmax); ymax = min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0: return
    xc, yc = xmin + bw / 2.0, ymin + bh / 2.0
    with open(filepath, "w") as f:
        f.write(f"3 {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

def process_images():
    mp_face_mesh = mp.solutions.face_mesh
    img_exts = (".png", ".jpg", ".jpeg", ".webp")
    
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(img_exts)]
    texture_files = [f for f in os.listdir(TEXTURE_DIR) if f.lower().endswith(img_exts)]
    
    if not input_files:
        print(f"'{INPUT_DIR}' フォルダに顔画像を入れてください。")
        return
    if not texture_files:
        print(f"'{TEXTURE_DIR}' フォルダに模様(テクスチャ)画像を入れてください。")
        return

    # テクスチャ画像を読み込んでおく
    textures = []
    for tf in texture_files:
        path = os.path.join(TEXTURE_DIR, tf)
        img = cv2.imread(path)
        if img is not None:
            textures.append((os.path.splitext(tf)[0], img))
            
    print(f"テクスチャ数: {len(textures)}枚")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        
        for filename in input_files:
            input_path = os.path.join(INPUT_DIR, filename)
            try:
                file_bytes = np.fromfile(input_path, np.uint8)
                img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_original is None: continue
            except: continue
            
            print(f"--- 処理開始: {filename} ---")
            h_img, w_img, _ = img_original.shape
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                print(f"スキップ: {filename} (FaceMesh検出不可)")
                continue

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # ランドマーク取得
                raw_face_pts = get_landmark_points(face_landmarks.landmark, FACE_OVAL_INDICES, w_img, h_img)
                left_eye_pts = get_landmark_points(face_landmarks.landmark, LEFT_EYE_INDICES, w_img, h_img)
                right_eye_pts = get_landmark_points(face_landmarks.landmark, RIGHT_EYE_INDICES, w_img, h_img)
                lips_pts = get_landmark_points(face_landmarks.landmark, LIPS_INDICES, w_img, h_img)
                left_brow_pts = get_landmark_points(face_landmarks.landmark, LEFT_EYEBROW_INDICES, w_img, h_img)
                right_brow_pts = get_landmark_points(face_landmarks.landmark, RIGHT_EYEBROW_INDICES, w_img, h_img)
                
                # BBox
                xs = [p[0] for p in raw_face_pts]
                ys = [p[1] for p in raw_face_pts]
                face_xmin, face_xmax = min(xs), max(xs)
                face_ymin, face_ymax = min(ys), max(ys)
                face_w, face_h = face_xmax - face_xmin, face_ymax - face_ymin

                face_pts = extend_forehead_points(raw_face_pts, FOREHEAD_INDICES, face_h, extension_ratio=0.15)
                
                margin = 0.5
                pad_x, pad_y = int(face_w * margin), int(face_h * margin)
                x1 = max(0, face_xmin - pad_x)
                y1 = max(0, face_ymin - pad_y)
                x2 = min(w_img, face_xmax + pad_x)
                y2 = min(h_img, face_ymax + pad_y)
                
                roi = img_original[y1:y2, x1:x2]
                if roi.size == 0: continue
                
                # マスク作成
                eye_ys = [p[1] for p in left_eye_pts + right_eye_pts]
                eye_level_y_rel = max(0, min(eye_ys) - y1)
                
                geo_mask = create_geometric_mask(
                    roi.shape, face_pts, left_eye_pts, right_eye_pts, lips_pts, 
                    left_brow_pts, right_brow_pts, (x1, y1)
                )
                skin_mask = create_advanced_skin_mask_with_position(
                    roi, face_landmarks, w_img, h_img, (x1, y1), eye_level_y_rel
                )
                final_mask = create_hybrid_mask(geo_mask, skin_mask, eye_level_y_rel)
                final_mask_normalized = final_mask.astype(np.float32) / 255.0

                # 各テクスチャを適用して保存
                for tex_name, tex_img in textures:
                    output_img = img_original.copy()
                    
                    # テクスチャ合成
                    textured_roi = apply_texture_pattern(roi, final_mask_normalized, tex_img)
                    output_img[y1:y2, x1:x2] = textured_roi
                    
                    base_name = os.path.splitext(filename)[0]
                    save_base = f"{base_name}_{tex_name}_{i}"
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{save_base}.jpg"), output_img)
                    
                    bbox = (face_xmin, face_ymin, face_xmax, face_ymax)
                    save_yolo_format(os.path.join(OUTPUT_DIR, f"{save_base}.txt"), w_img, h_img, bbox)
                    
                    print(f"    -> 保存: {save_base}")

    print("\nすべての処理が完了しました。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()