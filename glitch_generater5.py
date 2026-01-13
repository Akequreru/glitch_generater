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

# 追跡用ランドマーク（BBox計算用）
# 変形の影響を受けて座標がずれても追跡できるようにします
LM_INDICES_TRACK = {'top': 10, 'bottom': 152, 'left': 234, 'right': 454}

# 目の中心計算用（目の輪郭）
LM_EYES = {
    'left_eye': [33, 133, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155],
    'right_eye': [362, 263, 384, 385, 386, 387, 388, 466, 249, 390, 373, 374, 380, 381, 382]
}

def get_new_coords(src_pt, map_x, map_y, roi_offset):
    """変形後の座標をマップから逆算する"""
    src_x, src_y = src_pt
    off_x, off_y = roi_offset
    roi_src_x = src_x - off_x
    roi_src_y = src_y - off_y

    h, w = map_x.shape
    # ROI外の座標は移動しないとみなす
    if not (0 <= roi_src_x < w and 0 <= roi_src_y < h):
        return src_pt

    dist_map = (map_x - roi_src_x)**2 + (map_y - roi_src_y)**2
    min_idx = np.argmin(dist_map)
    dst_y_rel, dst_x_rel = np.unravel_index(min_idx, dist_map.shape)

    dst_x = dst_x_rel + off_x
    dst_y = dst_y_rel + off_y
    return (int(dst_x), int(dst_y))

# --- 変形ロジック1: 魚眼膨張 (従来のもの) ---
def apply_fisheye_stretch(roi, power=0.5):
    """中心を強く膨らませる魚眼効果"""
    h, w, _ = roi.shape
    cx, cy = w / 2.0, h / 2.0
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    vec_x = grid_x - cx
    vec_y = grid_y - cy
    distance = np.sqrt(vec_x**2 + vec_y**2)
    max_radius = min(w, h) / 2.0
    
    # 正規化
    r_norm = distance / max_radius
    mask_region = r_norm > 1.0 # 円の外側
    
    # 変形: r_src = r_dst ^ power (power < 1.0 で膨張)
    r_src_norm = np.power(r_norm, power)
    distance_src = r_src_norm * max_radius
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = distance_src / (distance + 1e-6)
        map_x = cx + vec_x * ratio
        map_y = cy + vec_y * ratio
    
    map_x[mask_region] = grid_x[mask_region]
    map_y[mask_region] = grid_y[mask_region]

    return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), map_x, map_y

# --- 変形ロジック2: 眼球突出 (Pop-out & Round Eyelids) ---
def apply_eyeball_pop(roi, pop_strength=2.0):
    """
    眼球が飛び出してまぶたが円形に歪む効果
    pop_strength: 1.5 〜 3.5 程度推奨 (値が大きいほど中心が巨大化し、まぶたが端へ追いやられる)
    """
    h, w, _ = roi.shape
    cx, cy = w / 2.0, h / 2.0
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    vec_x = grid_x - cx
    vec_y = grid_y - cy
    distance = np.sqrt(vec_x**2 + vec_y**2)
    max_radius = min(w, h) / 2.0
    
    # 正規化 (0.0 - 1.0)
    r_norm = distance / max_radius
    
    # 円形の外側は処理しない（マスク用）
    mask_region = r_norm > 1.0
    
    # --- 突出変形アルゴリズム ---
    # r_src = r_dst ^ k (k > 1.0)
    # これにより、中心付近の画素が広い範囲に引き伸ばされ（ズーム）、
    # 周辺（まぶた）の画素は円のフチに圧縮されて歪みます。
    # これが「まぶたが円形に引っ張られる」効果を生みます。
    r_src_norm = np.power(r_norm, pop_strength)
    
    distance_src = r_src_norm * max_radius
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 変換マップ作成
        scale = distance_src / (distance + 1e-6)
        map_x = cx + vec_x * scale
        map_y = cy + vec_y * scale
    
    # 範囲外は元の画像
    map_x[mask_region] = grid_x[mask_region]
    map_y[mask_region] = grid_y[mask_region]

    # BORDER_REFLECT_101 を使うことで、引き伸ばされた皮膚の質感が自然につながるようにする
    return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), map_x, map_y


def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    """YOLOフォーマット保存"""
    xmin, ymin, xmax, ymax = bbox_coords
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(img_w, xmax); ymax = min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    
    if bw <= 0 or bh <= 0: return
    xc, yc = xmin + bw / 2.0, ymin + bh / 2.0
    with open(filepath, "w") as f:
        f.write(f"0 {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

def process_images():
    mp_face_mesh = mp.solutions.face_mesh
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    
    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return

    # モード定義
    # fisheye: 従来の魚眼（中心膨張）
    # pop_out: 今回実装した「眼球飛び出し・まぶた円形化」
    modes = ['fisheye', 'pop_out']

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
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

            if not results.multi_face_landmarks:
                continue

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # 基準となる追跡点(4点)の取得
                pts_track_orig = {}
                for key, idx in LM_INDICES_TRACK.items():
                    lm = face_landmarks.landmark[idx]
                    pts_track_orig[key] = (int(lm.x * w_img), int(lm.y * h_img))
                
                # --- モードごとのループ ---
                for mode in modes:
                    output_img = img_original.copy()
                    current_pts_track = pts_track_orig.copy()

                    for eye_key, indices in LM_EYES.items():
                        eye_pts = [(int(face_landmarks.landmark[idx].x * w_img), int(face_landmarks.landmark[idx].y * h_img)) for idx in indices]
                        
                        xs, ys = [p[0] for p in eye_pts], [p[1] for p in eye_pts]
                        eye_cx, eye_cy = int(np.mean(xs)), int(np.mean(ys))
                        eye_w, eye_h = max(xs) - min(xs), max(ys) - min(ys)
                        
                        # ROIサイズ調整
                        # pop_outの場合、まぶたの「引きつり」を入れるためかなり広めに取る
                        scale_mult = 2.5 if mode == 'fisheye' else 3.2
                        roi_size = int(max(eye_w, eye_h) * scale_mult)
                        
                        x1 = max(0, eye_cx - roi_size // 2)
                        y1 = max(0, eye_cy - roi_size // 2)
                        x2 = min(w_img, eye_cx + roi_size // 2)
                        y2 = min(h_img, eye_cy + roi_size // 2)
                        
                        roi = output_img[y1:y2, x1:x2]
                        if roi.shape[0] == 0 or roi.shape[1] == 0: continue
                        
                        # --- 変形適用 ---
                        if mode == 'fisheye':
                            power = random.uniform(0.3, 0.6)
                            glitched_roi, map_x, map_y = apply_fisheye_stretch(roi, power=power)
                        else: # pop_out
                            # 強度 1.8 〜 3.0 で設定
                            # 値が高いほど「飛び出し感」と「まぶたの円形歪み」が強くなる
                            pop_str = random.uniform(1.8, 3.0)
                            glitched_roi, map_x, map_y = apply_eyeball_pop(roi, pop_strength=pop_str)
                        
                        # サイズが合う場合のみ書き戻し
                        if glitched_roi.shape == roi.shape:
                            output_img[y1:y2, x1:x2] = glitched_roi

                        # 座標追跡の更新
                        roi_offset = (x1, y1)
                        for key, pt in current_pts_track.items():
                            if x1 <= pt[0] < x2 and y1 <= pt[1] < y2:
                                new_pt = get_new_coords(pt, map_x, map_y, roi_offset)
                                current_pts_track[key] = new_pt

                    # --- 保存 ---
                    x_coords = [p[0] for p in current_pts_track.values()]
                    y_coords = [p[1] for p in current_pts_track.values()]
                    
                    new_xmin, new_ymin = min(x_coords), min(y_coords)
                    new_xmax, new_ymax = max(x_coords), max(y_coords)

                    name, ext = os.path.splitext(filename)
                    save_base = f"{name}_circle_{mode}_{i}"
                    img_path = os.path.join(OUTPUT_DIR, f"{save_base}{ext}")
                    txt_path = os.path.join(OUTPUT_DIR, f"{save_base}.txt")

                    is_success, buffer = cv2.imencode(ext, output_img)
                    if is_success:
                        with open(img_path, "wb") as f: f.write(buffer)
                        save_yolo_format(txt_path, w_img, h_img, (new_xmin, new_ymin, new_xmax, new_ymax))
                        print(f"    -> 保存完了: {save_base}")

    print("\nすべての処理が完了しました。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()