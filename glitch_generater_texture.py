import cv2
import mediapipe as mp
import numpy as np
import os
import random

# =========================================================
# 1. 設定 & 定数
# =========================================================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
TEXTURE_DIR = "texture_input"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEXTURE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# --- 肌検知・保護用パラメータ ---
COLOR_THRESHOLDS = {
    'h_tol': 20,
    's_tol_lower': 50,
    's_tol_upper': 80,
    'v_tol_lower': 120,
    'v_tol_upper': 100, 
    'cr_tol': 20, 
    'cb_tol': 20,
    'min_saturation_strict': 15,
    'highlight_threshold': 90 
}

# --- MediaPipe Face Mesh Indices ---
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356]
SKIN_SAMPLE_INDICES = [4, 234, 454, 152, 10]

LM_INDICES = {
    'face_oval': FACE_OVAL_INDICES,
    'bottom': 152, 
    'left_eye_contour': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    'right_eye_contour': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    'lips_contour': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17],
    'left_eyebrow': [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    'right_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'lips_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
}

FOREHEAD_EXTEND_RATIO = 0.07 

# =========================================================
# 2. 共通ヘルパー関数
# =========================================================

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    xmin, ymin, xmax, ymax = bbox_coords
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0: return
    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0
    n_xc, n_yc = xc / img_w, yc / img_h
    n_bw, n_bh = bw / img_w, bh / img_h
    with open(filepath, "w") as f:
        f.write(f"3 {n_xc:.6f} {n_yc:.6f} {n_bw:.6f} {n_bh:.6f}\n")

def extend_forehead_points(points, indices_to_extend, face_h, extension_ratio=0.15):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cy = sum(ys) / len(ys)
    min_y = min(ys)
    shift_amount = int(face_h * extension_ratio)
    modified_points = []
    for (x, y) in points:
        if y < cy and (y - min_y) < (face_h * 0.3):
            modified_points.append((x, max(0, y - shift_amount)))
        else:
            modified_points.append((x, y))
    return modified_points

# =========================================================
# 3. マスク生成ロジック
# =========================================================

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

    # 眉毛保護
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

# =========================================================
# 4. エフェクト適用ロジック (Gen 7, 8, 9)
# =========================================================

# --- Gen 7: Skin Color Change ---
def apply_skin_color_change(roi, mask, color_mode):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv_roi)

    if color_mode == 'red':
        h[:] = 175
        s_factor = random.uniform(1.5, 3.5)
        s = s * s_factor
        s = np.clip(s, 0, 255)
    elif color_mode == 'green':
        h[:] = 55
        s_factor = random.uniform(1.5, 3.5)
        s = np.maximum(s * s_factor, random.uniform(30, 60))
        s = np.clip(s, 0, 255)
    elif color_mode == 'blue':
        h[:] = 110 
        s_factor = random.uniform(1.5, 4.0)
        s = np.maximum(s * s_factor, random.uniform(40, 80))
        s = np.clip(s, 0, 255)
    elif color_mode == 'black':
        v_factor = random.uniform(0.05, 0.4)
        s_factor = random.uniform(0.0, 0.3)
        s = s * s_factor
        v = v * v_factor

    merged_hsv = cv2.merge([h, s, v])
    colored_roi = cv2.cvtColor(merged_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    roi_float = roi.astype(np.float32)
    colored_roi_float = colored_roi.astype(np.float32)
    
    # マスク適用
    mask_3ch = cv2.merge([mask, mask, mask])
    output = roi_float * (1.0 - mask_3ch) + colored_roi_float * mask_3ch
    return output.astype(np.uint8)

# --- Gen 8: Skin Noise ---
def apply_skin_noise(roi, mask, noise_mode):
    h, w, c = roi.shape
    roi_float = roi.astype(np.float32)
    noise_img = np.zeros_like(roi_float)
    if noise_mode == 'grain':
        sigma = random.uniform(20, 80)
        scale = random.uniform(0.15, 0.4) 
        small_h, small_w = int(h * scale), int(w * scale)
        if small_h < 1: small_h = 1
        if small_w < 1: small_w = 1
        gauss_small = np.random.normal(0, sigma, (small_h, small_w)).astype(np.float32)
        gauss = cv2.resize(gauss_small, (w, h), interpolation=cv2.INTER_LINEAR)
        noise_img[:, :, 0] = gauss
        noise_img[:, :, 1] = gauss
        noise_img[:, :, 2] = gauss
    elif noise_mode == 'digital':
        sigma = random.uniform(30, 100)
        scale = random.uniform(0.05, 0.25)
        small_h, small_w = int(h * scale), int(w * scale)
        if small_h < 1: small_h = 1
        if small_w < 1: small_w = 1
        noise_small = np.random.normal(0, sigma, (small_h, small_w, c)).astype(np.float32)
        noise_img = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_NEAREST)
    noisy_roi = cv2.add(roi_float, noise_img)
    noisy_roi = np.clip(noisy_roi, 0, 255).astype(np.uint8)
    mask_3ch = cv2.merge([mask, mask, mask])
    output = noisy_roi.astype(np.float32) * mask_3ch + roi_float * (1.0 - mask_3ch)
    return output.astype(np.uint8)

# --- Gen 9: Texture ---
def apply_texture_pattern(roi, mask, texture_img):
    h, w = roi.shape[:2]
    th, tw = texture_img.shape[:2]
    if th < h or tw < w:
        scale = max(h / th, w / tw) * 1.5 
        texture_img = cv2.resize(texture_img, None, fx=scale, fy=scale)
        th, tw = texture_img.shape[:2]
    x_start = random.randint(0, tw - w)
    y_start = random.randint(0, th - h)
    crop = texture_img[y_start:y_start+h, x_start:x_start+w]
    gray_tex = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex_mean = np.mean(gray_tex)
    tex_diff = gray_tex - tex_mean
    strength = random.uniform(0.5, 1.5)
    tex_diff = tex_diff * strength
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv_roi)
    v_new = v_ch + tex_diff
    v_new = np.clip(v_new, 0, 255)
    hsv_new = cv2.merge([h_ch, s_ch, v_new])
    textured_roi = cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
    mask_3ch = cv2.merge([mask, mask, mask])
    roi_float = roi.astype(np.float32)
    tex_float = textured_roi.astype(np.float32)
    output = tex_float * mask_3ch + roi_float * (1.0 - mask_3ch)
    return output.astype(np.uint8)

# =========================================================
# メイン処理
# =========================================================

def process_images():
    mp_face_mesh = mp.solutions.face_mesh
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    # テクスチャ読み込み
    textures = []
    tex_files = [f for f in os.listdir(TEXTURE_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    for tf in tex_files:
        img = cv2.imread(os.path.join(TEXTURE_DIR, tf))
        if img is not None: textures.append((os.path.splitext(tf)[0], img))

    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return
    print(f"検出ファイル数: {len(files)} 枚 / テクスチャ数: {len(textures)} 枚")

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for filename in files:
            input_path = os.path.join(INPUT_DIR, filename)
            try:
                file_bytes = np.fromfile(input_path, np.uint8)
                img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_original is None: continue
            except Exception: continue
            
            print(f"\n--- 処理中: {filename} ---")
            h_img, w_img, _ = img_original.shape
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks: continue

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # ----------------------------------------------------
                # 1. 座標取得 & ROI計算
                # ----------------------------------------------------
                pts = {}
                for key, raw_val in LM_INDICES.items():
                    if isinstance(raw_val, int):
                        indices_list = [raw_val]
                    else:
                        indices_list = raw_val
                    pts[key] = [(int(face_landmarks.landmark[idx].x * w_img), int(face_landmarks.landmark[idx].y * h_img)) for idx in indices_list]
                
                xs_all = [lm.x for lm in face_landmarks.landmark]
                ys_all = [lm.y for lm in face_landmarks.landmark]
                face_xmin, face_xmax = int(min(xs_all) * w_img), int(max(xs_all) * w_img)
                raw_ymin, raw_ymax = int(min(ys_all) * h_img), int(max(ys_all) * h_img)
                face_w = face_xmax - face_xmin
                face_h_raw = raw_ymax - raw_ymin
                
                margin_up = int(face_h_raw * FOREHEAD_EXTEND_RATIO)
                face_ymin = max(0, raw_ymin - margin_up)
                face_ymax = raw_ymax
                face_h = face_ymax - face_ymin
                
                base_name = os.path.splitext(filename)[0]
                _, ext = os.path.splitext(filename)

                # ----------------------------------------------------
                # 2. ROI切り出し & マスク作成
                # ----------------------------------------------------
                face_pts_ext = extend_forehead_points(pts['face_oval'], FOREHEAD_INDICES, face_h, 0.15)
                px, py = int(face_w*0.5), int(face_h*0.5)
                sx1, sy1 = max(0, face_xmin-px), max(0, face_ymin-py)
                sx2, sy2 = min(w_img, face_xmax+px), min(h_img, face_ymax+py)
                roi_skin = img_original[sy1:sy2, sx1:sx2]
                
                if roi_skin.size > 0:
                    eye_ys = [p[1] for p in pts['left_eye_contour'] + pts['right_eye_contour']]
                    if eye_ys:
                        eye_lvl = max(0, min(eye_ys) - sy1)
                    else:
                        eye_lvl = roi_skin.shape[0] // 3
                    
                    geo_mask = create_geometric_mask(roi_skin.shape, face_pts_ext, pts['left_eye_contour'], pts['right_eye_contour'], pts['lips_inner'], pts['left_eyebrow'], pts['right_eyebrow'], (sx1, sy1))
                    skin_mask = create_advanced_skin_mask_with_position(roi_skin, face_landmarks, w_img, h_img, (sx1, sy1), eye_lvl)
                    final_mask = create_hybrid_mask(geo_mask, skin_mask, eye_lvl)
                    final_mask_norm = final_mask.astype(np.float32) / 255.0

                    # ----------------------------------------------------
                    # 3. エフェクト適用ループ
                    # ----------------------------------------------------

                    # === Gen 7: 肌色変更 (Red, Green, Blue, Black) ===
                    for color in ['black', 'red', 'green', 'blue']:
                        out = img_original.copy()
                        out[sy1:sy2, sx1:sx2] = apply_skin_color_change(roi_skin, final_mask_norm, color)
                        sname = f"{base_name}_{color}_{i}"
                        cv2.imencode(ext, out)[1].tofile(os.path.join(OUTPUT_DIR, sname+ext))
                        save_yolo_format(os.path.join(OUTPUT_DIR, sname+".txt"), w_img, h_img, (face_xmin, face_ymin, face_xmax, face_ymax))
                        print(f"    -> 生成(Color): {sname}")

                    # === Gen 8: ノイズ (Grain, Digital) ===
                    for mode in ['grain', 'digital']:
                        out = img_original.copy()
                        out[sy1:sy2, sx1:sx2] = apply_skin_noise(roi_skin, final_mask_norm, mode)
                        sname = f"{base_name}_{mode}_{i}"
                        cv2.imencode(ext, out)[1].tofile(os.path.join(OUTPUT_DIR, sname+ext))
                        save_yolo_format(os.path.join(OUTPUT_DIR, sname+".txt"), w_img, h_img, (face_xmin, face_ymin, face_xmax, face_ymax))
                        print(f"    -> 生成(Noise): {sname}")

                    # === Gen 9: テクスチャ (Texture) ===
                    for tname, timg in textures:
                        out = img_original.copy()
                        out[sy1:sy2, sx1:sx2] = apply_texture_pattern(roi_skin, final_mask_norm, timg)
                        sname = f"{base_name}_{tname}_{i}"
                        cv2.imencode(ext, out)[1].tofile(os.path.join(OUTPUT_DIR, sname+ext))
                        save_yolo_format(os.path.join(OUTPUT_DIR, sname+".txt"), w_img, h_img, (face_xmin, face_ymin, face_xmax, face_ymax))
                        print(f"    -> 生成(Texture): {sname}")

    print("\n全工程完了。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()