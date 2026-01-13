import cv2
import mediapipe as mp
import numpy as np
import os
import random
import math

# --- 設定 ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 画像拡張子
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 生え際アノテーションの補正値
FOREHEAD_EXTEND_RATIO = 0.07 

# --- MediaPipe Face Mesh 定義 ---
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
    54, 103, 67, 109
]

LM_INDICES = {
    'face_oval': FACE_OVAL_INDICES,
    'bottom': 152, 
    'left_eye_top_curve': [33, 246, 161, 160, 159, 158, 157, 173, 133],
    'left_eye_bottom_curve': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'right_eye_top_curve': [362, 398, 384, 385, 386, 387, 388, 466, 263],
    'right_eye_bottom_curve': [362, 382, 381, 380, 374, 373, 390, 249, 263],
    'left_eye_contour': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    'right_eye_contour': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    'lips_contour': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17]
}

# =========================================================
# 共通ユーティリティ関数
# =========================================================

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    """YOLOフォーマット保存"""
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
        f.write(f"0 {n_xc:.6f} {n_yc:.6f} {n_bw:.6f} {n_bh:.6f}\n")

def get_roi_rect(center_x, center_y, width, height, margin, img_w, img_h):
    """ROI座標を計算して画像範囲内にクリップする"""
    roi_size_w = width * (1 + margin)
    roi_size_h = height * (1 + margin)
    roi_size = max(roi_size_w, roi_size_h)

    x1 = max(0, int(center_x - roi_size/2))
    y1 = max(0, int(center_y - roi_size/2))
    x2 = min(img_w, int(center_x + roi_size/2))
    y2 = min(img_h, int(center_y + roi_size/2))
    return x1, y1, x2, y2

def get_new_coords(src_pt, map_x, map_y, roi_offset):
    """変形後の座標をマップから逆算する (Gen 6 logic)"""
    src_x, src_y = src_pt
    off_x, off_y = roi_offset
    roi_src_x = src_x - off_x
    roi_src_y = src_y - off_y
    h, w = map_x.shape
    if not (0 <= roi_src_x < w and 0 <= roi_src_y < h):
        return src_pt # ROI外なら移動なし
    
    # マップ上で最も近いピクセルを探す（逆写像近似）
    dist_map = (map_x - roi_src_x)**2 + (map_y - roi_src_y)**2
    min_idx = np.argmin(dist_map)
    dst_y_rel, dst_x_rel = np.unravel_index(min_idx, dist_map.shape)
    
    return (int(dst_x_rel + off_x), int(dst_y_rel + off_y))

def update_face_oval_points(face_oval_pts, transformation_maps):
    """
    複数の変形マップを適用して顔の輪郭点を更新する
    transformation_maps: list of (roi_offset, map_x, map_y)
    """
    current_pts = list(face_oval_pts) # コピーを作成
    
    for roi_offset, map_x, map_y in transformation_maps:
        if map_x is None or map_y is None: continue
        
        updated_pts = []
        x1, y1 = roi_offset
        h, w = map_x.shape
        x2, y2 = x1 + w, y1 + h

        for pt in current_pts:
            # 点がこのROIに含まれる場合のみ座標更新を試みる
            if x1 <= pt[0] < x2 and y1 <= pt[1] < y2:
                updated_pts.append(get_new_coords(pt, map_x, map_y, roi_offset))
            else:
                updated_pts.append(pt)
        current_pts = updated_pts

    return current_pts

def calculate_tight_bbox(points, img_w, img_h, extend_ratio=FOREHEAD_EXTEND_RATIO):
    """点群からBBoxを計算し、生え際補正を行う"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    h_curr = ymax - ymin
    margin_up = int(h_curr * extend_ratio)
    final_ymin = max(0, ymin - margin_up)
    
    return (xmin, final_ymin, xmax, ymax)

# =========================================================
# 統合された保存・実行ラッパー
# =========================================================

def apply_and_save(img_original, filename, face_idx, suffix, 
                   roi_list, effect_func, effect_kwargs, 
                   face_oval_pts, w_img, h_img):
    """
    共通処理: ROI切り出し -> エフェクト -> 座標追跡 -> 保存
    roi_list: list of (x1, y1, x2, y2)  (複数ROIに対応)
    """

    base_name, ext = os.path.splitext(filename)
    output_img = img_original.copy()
    transformation_maps = [] # (roi_offset, map_x, map_y) を保持

    # 各ROIに対してエフェクト適用
    for (x1, y1, x2, y2) in roi_list:
        roi_src = output_img[y1:y2, x1:x2]
        if roi_src.size == 0: continue
        
        # エフェクト関数呼び出し
        # 戻り値は (processed_roi, map_x, map_y) を期待
        res = effect_func(roi_src, **effect_kwargs)
        
        # 戻り値の形式に合わせて処理
        if len(res) == 3:
            processed_roi, map_x, map_y = res
        else:
            processed_roi = res[0]
            map_x, map_y = None, None

        # 画像合成
        output_img[y1:y2, x1:x2] = processed_roi
        
        # 座標変換用マップを保存
        if map_x is not None:
            transformation_maps.append(((x1, y1), map_x, map_y))

    # Gen 6 Logic: 顔の輪郭点を追跡して新しいBBoxを計算
    if transformation_maps:
        new_oval_pts = update_face_oval_points(face_oval_pts, transformation_maps)
    else:
        new_oval_pts = face_oval_pts # 変形なしの場合はそのまま

    final_bbox = calculate_tight_bbox(new_oval_pts, w_img, h_img)

    # 保存
    save_name = f"{base_name}_{suffix}_{face_idx}"
    # img_path = os.path.join(OUTPUT_DIR, save_name + os.path.splitext(base_name)[1]) # 元の拡張子維持が難しいためbase_name依存修正
    # # 拡張子の処理が少し複雑なので簡易化
    # ext = os.path.splitext(base_name)[1]
    save_full_path = os.path.join(OUTPUT_DIR, save_name + ext)
    
    cv2.imencode(ext, output_img)[1].tofile(save_full_path)
    save_yolo_format(os.path.join(OUTPUT_DIR, save_name + ".txt"), w_img, h_img, final_bbox)
    print(f"    -> 生成: {save_name}")


# =========================================================
# エフェクトロジック (マップを返すように改修)
# =========================================================

def apply_vertical_melt(roi, face_h, power, start_rel, level_rel):
    h, w, _ = roi.shape
    y_indices = np.arange(h, dtype=np.float32)
    face_h = max(face_h, 1.0)
    normalized_y = y_indices / face_h 
    slope = 1.0 / (level_rel - start_rel + 1e-6)
    curve_y = np.clip((normalized_y - start_rel) * slope, 0.0, 1.0) ** 2 
    x_indices = np.linspace(-1, 1, w, dtype=np.float32)
    curve_x = np.clip(1.0 - (np.abs(x_indices) ** 4), 0.0, 1.0)
    
    melt_map = power * (curve_y[:, None] * curve_x[None, :])
    noise = cv2.GaussianBlur(np.random.rand(1, w).astype(np.float32) * 0.1, (0, 0), sigmaX=5)
    noise_mask = np.clip(1.0 - (normalized_y - start_rel) * slope, 0.0, 1.0)
    melt_map += noise * noise_mask[:, None] * curve_x[None, :]
    melt_map = np.clip(melt_map, 0.0, 0.99)
    steps = 1.0 - melt_map
    map_y = np.cumsum(steps, axis=0).astype(np.float32)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    return cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), map_x, map_y

def apply_squeeze_pinch(roi, face_h, power):
    h, w, _ = roi.shape
    cx, cy = w / 2.0, face_h / 2.0 
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = grid_x - cx, grid_y - cy
    radius = np.sqrt(dx**2 + dy**2)
    max_radius = np.sqrt(cx**2 + cy**2) * 1.5 + 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.maximum(0.0, 1.0 - radius / max_radius)
        pinch_factor = 1.0 - power * (factor ** 2)
        new_x = cx + dx / pinch_factor
        new_y = cy + dy / pinch_factor
        mask = pinch_factor <= 0.01
        new_x[mask] = cx + dx[mask] * 100
        new_y[mask] = cy + dy[mask] * 100
    map_x, map_y = new_x.astype(np.float32), new_y.astype(np.float32)
    return cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), map_x, map_y

def apply_eye_morph(roi, roi_offset, pts_dict, mode='orig_open'):
    """
    Gen 3用: 白目化(Pixel)と開眼(Warp)を一括処理する関数
    mode: 'blank_only', 'blank_open', 'orig_open'
    """
    h, w = roi.shape[:2]
    gx1, gy1 = roi_offset
    
    # ローカル座標変換ヘルパー
    def to_loc(pl): return [(p[0]-gx1, p[1]-gy1) for p in pl]

    # --- 1. 白目化処理 (Pixel Manipulation) ---
    if 'blank' in mode:
        # 輪郭の取得
        cnt_l = pts_dict['left_eye_top_curve'] + pts_dict['left_eye_bottom_curve'][::-1]
        cnt_r = pts_dict['right_eye_top_curve'] + pts_dict['right_eye_bottom_curve'][::-1]
        
        # マスク作成
        mask_l = create_single_eye_mask(h, w, to_loc(cnt_l), (0,0))
        mask_r = create_single_eye_mask(h, w, to_loc(cnt_r), (0,0))
        col_l = get_sclera_color(roi, mask_l)
        col_r = get_sclera_color(roi, mask_r)
        
        # 塗りつぶし
        blur = (5, 5)
        m_l = cv2.cvtColor(cv2.GaussianBlur(mask_l, blur, 0), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        m_r = cv2.cvtColor(cv2.GaussianBlur(mask_r, blur, 0), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        fill_l = np.full_like(roi, col_l, dtype=np.float32)
        fill_r = np.full_like(roi, col_r, dtype=np.float32)
        
        working_roi = roi.astype(np.float32)
        working_roi = working_roi * (1 - m_l) + fill_l * m_l
        working_roi = working_roi * (1 - m_r) + fill_r * m_r
        working_roi = np.clip(working_roi, 0, 255).astype(np.uint8)
    else:
        working_roi = roi.copy()

    # --- 2. 開眼変形 (Geometric Warp) ---
    if 'open' in mode:
        p_l_t = to_loc(pts_dict['left_eye_top_curve'])
        p_l_b = to_loc(pts_dict['left_eye_bottom_curve'])
        p_r_t = to_loc(pts_dict['right_eye_top_curve'])
        p_r_b = to_loc(pts_dict['right_eye_bottom_curve'])
        
        # ランダムパラメータ
        bp = random.uniform(0.3, 1.1)
        bs = random.uniform(1.5, 3.0)
        pl, pr = bp * random.uniform(0.8, 1.2), bp * random.uniform(0.8, 1.2)
        sl, sr = bs * random.uniform(0.9, 1.1), bs * random.uniform(0.9, 1.1)

        # マップ作成用グリッド
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

        # 左目変形
        temp_img, mx1, my1 = apply_eyelid_movement(working_roi, p_l_t, p_l_b, pl, sl)
        
        # 右目変形 (temp_imgに対して適用)
        final_img, mx2, my2 = apply_eyelid_movement(temp_img, p_r_t, p_r_b, pr, sr)
        
        # マップ合成 (BBox追跡精度向上のため、左右の変形量を足し合わせる)
        # diff_1 = mx1 - grid_x, diff_2 = mx2 - grid_x
        # 左右の目が離れている前提で、変形量を単純加算して合成マップを作成
        diff_x_total = (mx1 - grid_x) + (mx2 - grid_x)
        diff_y_total = (my1 - grid_y) + (my2 - grid_y)
        
        merged_map_x = grid_x + diff_x_total
        merged_map_y = grid_y + diff_y_total
        
        return final_img, merged_map_x, merged_map_y

    else:
        # 変形なし (Blank Onlyの場合)
        return working_roi, None, None

def apply_eyelid_movement(roi, top_pts, bottom_pts, power=0.6, sigma_scale=2.0):
    # Gen 3用: マップを返すように変更
    h, w = roi.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
    
    all_pts = np.array(top_pts + bottom_pts)
    if len(all_pts) == 0: return roi, map_x, map_y
    
    center_y = np.mean(all_pts[:, 1])
    eye_h = np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])
    if eye_h <= 0: eye_h = 1.0
    
    delta_y = map_y - center_y
    sigma_y = eye_h * sigma_scale 
    weight = np.exp(-(delta_y**2 / (2 * sigma_y**2)))
    scaling = 1.0 - (power * weight)
    map_y_new = center_y + delta_y * scaling
    
    return cv2.remap(roi, map_x, map_y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), map_x, map_y_new

def apply_omni_directional_stretch(roi, contour_points, roi_offset):
    # Gen 4用: マップを返すように変更
    h, w, _ = roi.shape
    off_x, off_y = roi_offset
    rel_contour = np.array([(p[0] - off_x, p[1] - off_y) for p in contour_points], dtype=np.int32)
    
    # 輪郭拡大
    rel_contour = expand_contour(rel_contour, scale=1.8)
    
    M = cv2.moments(rel_contour)
    if M['m00'] == 0: 
        # 変形なし
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        return roi, grid_x, grid_y

    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    _, _, w_cnt, h_cnt = cv2.boundingRect(rel_contour)
    
    grid_x, grid_y = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))
    vec_x, vec_y = grid_x - cx, grid_y - cy
    distance = np.sqrt(vec_x**2 + vec_y**2) + 1e-6
    unit_x, unit_y = vec_x / distance, vec_y / distance

    stretch_dist_x = w_cnt * 0.6
    stretch_dist_y = h_cnt * 0.6
    map_x = grid_x + unit_x * stretch_dist_x
    map_y = grid_y + unit_y * stretch_dist_y

    # マスク処理
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [rel_contour], 1.0)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_3ch = np.dstack([mask]*3)
    
    warped_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    final_roi = warped_roi * mask_3ch + roi.astype(np.float32) * (1.0 - mask_3ch)
    
    return final_roi.astype(np.uint8), map_x, map_y

def apply_fisheye_stretch(roi, power=0.5):
    h, w, _ = roi.shape
    cx, cy = w / 2.0, h / 2.0
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    vec_x, vec_y = grid_x - cx, grid_y - cy
    distance = np.sqrt(vec_x**2 + vec_y**2)
    max_radius = min(w, h) / 2.0
    
    r_norm = distance / max_radius
    mask_region = r_norm > 1.0
    r_src_norm = np.power(r_norm, power)
    distance_src = r_src_norm * max_radius
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = distance_src / (distance + 1e-6)
        map_x = cx + vec_x * ratio
        map_y = cy + vec_y * ratio
    
    map_x[mask_region] = grid_x[mask_region]
    map_y[mask_region] = grid_y[mask_region]
    return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), map_x, map_y

def apply_eyeball_pop(roi, pop_strength=2.0):
    h, w, _ = roi.shape
    cx, cy = w / 2.0, h / 2.0
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    vec_x, vec_y = grid_x - cx, grid_y - cy
    distance = np.sqrt(vec_x**2 + vec_y**2)
    max_radius = min(w, h) / 2.0
    r_norm = distance / max_radius
    mask_region = r_norm > 1.0
    r_src_norm = np.power(r_norm, pop_strength)
    distance_src = r_src_norm * max_radius
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = distance_src / (distance + 1e-6)
        map_x = cx + vec_x * scale
        map_y = cy + vec_y * scale
    
    map_x[mask_region] = grid_x[mask_region]
    map_y[mask_region] = grid_y[mask_region]
    return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), map_x, map_y

def apply_random_suction(roi, target_point, face_size, power):
    h, w, _ = roi.shape
    cx, cy = target_point
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = grid_x - cx, grid_y - cy
    radius = np.sqrt(dx**2 + dy**2)
    max_radius = face_size * 2.0 + 1e-6

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.maximum(0.0, 1.0 - radius / max_radius)
        pinch_factor = 1.0 - power * (factor ** 2)
        new_x = cx + dx / pinch_factor
        new_y = cy + dy / pinch_factor
        mask = pinch_factor <= 0.01
        new_x[mask] = cx + dx[mask] * 100
        new_y[mask] = cy + dy[mask] * 100

    map_x, map_y = new_x.astype(np.float32), new_y.astype(np.float32)
    distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return distorted, map_x, map_y

def create_single_eye_mask(h, w, contour_pts, roi_offset):
    mask = np.zeros((h, w), dtype=np.uint8)
    off_x, off_y = roi_offset
    pts_rel = [(pt[0] - off_x, pt[1] - off_y) for pt in contour_pts]
    if not pts_rel: return mask
    cv2.fillPoly(mask, [np.array(pts_rel, dtype=np.int32)], 255)
    return mask

def get_sclera_color(roi, mask):
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

def expand_contour(contour, scale=1.8):
    if len(contour) == 0: return contour
    M = cv2.moments(contour)
    if M['m00'] == 0: return contour
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cnt_norm = contour - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_new = cnt_scaled + [cx, cy]
    return cnt_new.astype(np.int32)

# =========================================================
# メイン処理
# =========================================================

def process_images():
    mp_face_mesh = mp.solutions.face_mesh
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return
    print(f"検出ファイル数: {len(files)} 枚")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        
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
            
            if not results.multi_face_landmarks:
                print("  -> 顔検出なしのためスキップ")
                continue

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # 座標取得
                pts = {}
                for key, indices in LM_INDICES.items():
                    if isinstance(indices, int):
                        lm = face_landmarks.landmark[indices]
                        pts[key] = (int(lm.x * w_img), int(lm.y * h_img))
                    else:
                        pts[key] = [(int(face_landmarks.landmark[idx].x * w_img), int(face_landmarks.landmark[idx].y * h_img)) for idx in indices]
                
                # 基本となる顔矩形
                xs_all = [lm.x for lm in face_landmarks.landmark]
                ys_all = [lm.y for lm in face_landmarks.landmark]
                face_xmin, face_xmax = int(min(xs_all) * w_img), int(max(xs_all) * w_img)
                raw_ymin, raw_ymax = int(min(ys_all) * h_img), int(max(ys_all) * h_img)
                face_w = face_xmax - face_xmin
                face_h = raw_ymax - raw_ymin
                
                # アノテーション追跡用（全Gen共通）
                face_oval_points = pts['face_oval']

                # ----------------------------------------------------
                # Generator 2 (Melt & Squeeze)
                # ----------------------------------------------------
                gen2_effects = ['melt', 'squeeze']
                margin_x, margin_top, margin_bottom = 0.1, 0.5, 2.5
                x_m = int(face_xmin - face_w * margin_x)
                y_m = int(max(0, raw_ymin - face_h * margin_top)) # 生え際補正はあとでやるのでここでは広く取る
                w_m = int(face_w * (1 + margin_x * 2))
                h_m = int(face_h * (1 + margin_top + margin_bottom))
                x1, y1 = max(0, x_m), max(0, y_m)
                x2, y2 = min(w_img, x_m + w_m), min(h_img, y_m + h_m)
                roi_gen2 = [(x1, y1, x2, y2)]
                
                face_chin_y_rel = pts['bottom'][1] - y1

                for eff in gen2_effects:
                    rnd_pow = random.uniform(0.4, 0.7) if eff == 'melt' else random.uniform(0.4, 0.95)
                    kwargs = {'face_h': face_chin_y_rel, 'power': rnd_pow}
                    
                    func = apply_squeeze_pinch
                    if eff == 'melt':
                        func = apply_vertical_melt
                        v1, v2 = random.uniform(0.4, 0.95), random.uniform(0.4, 0.95)
                        kwargs.update({'start_rel': min(v1, v2), 'level_rel': max(v1, v2)})
                    
                    apply_and_save(img_original, filename, i, eff,
                                    roi_gen2, func, kwargs, face_oval_points, w_img, h_img)

# ----------------------------------------------------
                # 3. Generator 3 機能 (Blank Eyes & Open Eyes)
                # ----------------------------------------------------
                roi_cx, roi_cy = (face_xmin + face_xmax) // 2, (raw_ymin + raw_ymax) // 2
                # 顔全体を含むROI (margin 0.4 = 1.4倍)
                gx1, gy1, gx2, gy2 = get_roi_rect(roi_cx, roi_cy, face_w, face_h, 0.4, w_img, h_img)
                roi_g3 = [(gx1, gy1, gx2, gy2)]
                
                # 共通引数
                g3_kwargs = {'pts_dict': pts, 'roi_offset': (gx1, gy1)}

                # 3-1. Blank Only (変形なし・白目のみ)
                # マップはNoneが返るため、BBoxは顔全体のものがそのまま使用されます
                apply_and_save(img_original, filename, i, "blank_only",
                               roi_g3, apply_eye_morph, {**g3_kwargs, 'mode': 'blank_only'},
                               face_oval_points, w_img, h_img)

                # 3-2. Blank Open (白目 + 開眼変形)
                apply_and_save(img_original, filename, i, "blank_open",
                               roi_g3, apply_eye_morph, {**g3_kwargs, 'mode': 'blank_open'},
                               face_oval_points, w_img, h_img)

                # 3-3. Orig Open (元画像 + 開眼変形)
                apply_and_save(img_original, filename, i, "orig_open",
                               roi_g3, apply_eye_morph, {**g3_kwargs, 'mode': 'orig_open'},
                               face_oval_points, w_img, h_img)

# ----------------------------------------------------
                # 4. Generator 4 (Faceless)
                # ----------------------------------------------------
                targets = [("eyes_hidden", ['left_eye_contour', 'right_eye_contour']),
                           ("mouth_hidden", ['lips_contour'])]

                for suffix, keys in targets:
                    target_pts = []
                    for k in keys: target_pts.extend(pts[k])
                    
                    # --- 修正: エフェクト崩れを防ぐため、ROIは「対象パーツ基準」に戻す ---
                    txs = [p[0] for p in target_pts]
                    tys = [p[1] for p in target_pts]
                    t_w, t_h = max(txs)-min(txs), max(tys)-min(tys)
                    cx, cy = int(np.mean(txs)), int(np.mean(tys))
                    
                    # マージン1.0 (パーツの倍のサイズ) で切り出し
                    hx1, hy1, hx2, hy2 = get_roi_rect(cx, cy, t_w, t_h, 1.0, w_img, h_img)
                    
                    # 複数パーツ処理用ラッパー
                    def apply_multi_omni(roi_in, key_list, pts_dict, offset):
                        r = roi_in.copy()
                        # マップは返さず(None)、画像のみ返す
                        # -> apply_and_save はマップがNoneの場合、元の顔輪郭(face_oval_points)を
                        #    そのまま使ってBBoxを作るため、結果的に「顔全体のアノテーション」になります
                        for k in key_list:
                            r, _, _ = apply_omni_directional_stretch(r, pts_dict[k], offset)
                        return r, None, None

                    apply_and_save(img_original, filename, i, suffix,
                                   [(hx1, hy1, hx2, hy2)], apply_multi_omni, 
                                   {'key_list': keys, 'pts_dict': pts, 'offset': (hx1, hy1)},
                                   face_oval_points, w_img, h_img)

                # ----------------------------------------------------
                # Generator 5 (Fisheye & Pop-out)
                # ----------------------------------------------------
                # 左右の目を別々のROIとしてリスト化し、一括処理させる
                gen5_modes = ['fisheye', 'pop_out']
                eye_keys = ['left_eye_contour', 'right_eye_contour']
                
                for mode in gen5_modes:
                    rois = []
                    
                    for k in eye_keys:
                        e_pts = pts[k]
                        xs, ys = [p[0] for p in e_pts], [p[1] for p in e_pts]
                        ecx, ecy = int(np.mean(xs)), int(np.mean(ys))
                        ew, eh = max(xs)-min(xs), max(ys)-min(ys)
                        scale = 2.5 if mode == 'fisheye' else 3.2
                        size = int(max(ew, eh) * scale)
                        ex1, ey1, ex2, ey2 = get_roi_rect(ecx, ecy, size, size, 0, w_img, h_img)
                        rois.append((ex1, ey1, ex2, ey2))
                    
                    target_func = apply_fisheye_stretch if mode == 'fisheye' else apply_eyeball_pop
                    kw = {'power': random.uniform(0.3, 0.6)} if mode == 'fisheye' else {'pop_strength': random.uniform(1.8, 3.0)}
                    
                    apply_and_save(img_original, filename, i, f"circle_{mode}",
               rois, target_func, kw, face_oval_points, w_img, h_img)

                # ----------------------------------------------------
                # Generator 6 (Random Suction)
                # ----------------------------------------------------
                margin_g6 = 0.8
                roi_len = int(max(face_w, face_h) * (1 + margin_g6))
                face_cx, face_cy = (face_xmin + face_xmax)//2, (raw_ymin + raw_ymax)//2
                sx1, sy1, sx2, sy2 = get_roi_rect(face_cx, face_cy, roi_len, roi_len, 0, w_img, h_img)
                
                # ターゲット決定
                angle_rnd = random.uniform(0, 2 * math.pi)
                dist_rnd = max(face_w, face_h) * random.uniform(0.0, 0.8)
                target_abs_x = face_cx + dist_rnd * math.cos(angle_rnd)
                target_abs_y = face_cy + dist_rnd * math.sin(angle_rnd)
                target_pt = (target_abs_x - sx1, target_abs_y - sy1)

                apply_and_save(img_original, filename, i, "warp_squeeze",
               [(sx1, sy1, sx2, sy2)], apply_random_suction,
               {'target_point': target_pt, 'face_size': max(face_w, face_h), 'power': random.uniform(0.6, 0.95)},
               face_oval_points, w_img, h_img)

    print("\n全工程完了。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()