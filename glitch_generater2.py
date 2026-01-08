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

# 固定設定
MELT_NOISE_SCALE = 0.1

# MediaPipe Face Meshのランドマークインデックス
# 10: おでこ(生え際), 152: 顎の先, 234: 左の頬, 454: 右の頬
LM_INDICES = {'top': 10, 'bottom': 152, 'left': 234, 'right': 454}

def get_new_coords(src_pt, map_x, map_y, roi_offset):
    """変形後の座標をマップから逆算する"""
    src_x, src_y = src_pt
    off_x, off_y = roi_offset
    roi_src_x = src_x - off_x
    roi_src_y = src_y - off_y

    # 探索範囲を変形後のROI内に限定
    dist_map = (map_x - roi_src_x)**2 + (map_y - roi_src_y)**2
    min_idx = np.argmin(dist_map)
    dst_y_rel, dst_x_rel = np.unravel_index(min_idx, dist_map.shape)

    dst_x = dst_x_rel + off_x
    dst_y = dst_y_rel + off_y
    return (int(dst_x), int(dst_y))

def apply_vertical_melt(roi, face_h, power, start_rel, level_rel):
    """縦溶けエフェクト（マップも返す）"""
    h, w, _ = roi.shape
    y_indices = np.arange(h, dtype=np.float32)
    
    # ゼロ除算防止
    if face_h <= 0: face_h = 1.0
    normalized_y = y_indices / face_h 
    
    slope = 1.0 / (level_rel - start_rel + 1e-6) # ゼロ除算防止
    curve_y = np.clip((normalized_y - start_rel) * slope, 0.0, 1.0) ** 2 
    
    x_indices = np.linspace(-1, 1, w, dtype=np.float32)
    curve_x = np.clip(1.0 - (np.abs(x_indices) ** 4), 0.0, 1.0)

    melt_map = power * (curve_y[:, None] * curve_x[None, :])
    
    noise = cv2.GaussianBlur(np.random.rand(1, w).astype(np.float32) * MELT_NOISE_SCALE, (0, 0), sigmaX=5)
    noise_mask = np.clip(1.0 - (normalized_y - start_rel) * slope, 0.0, 1.0)
    melt_map += noise * noise_mask[:, None] * curve_x[None, :]
    melt_map = np.clip(melt_map, 0.0, 0.99)

    steps = 1.0 - melt_map
    map_y = np.cumsum(steps, axis=0).astype(np.float32)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))

    distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return distorted, map_x, map_y

def apply_squeeze_pinch(roi, face_h, power):
    """縮みエフェクト（マップも返す）"""
    h, w, _ = roi.shape
    cx, cy = w / 2.0, face_h / 2.0 
    
    # 高速化のためグリッド計算
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    radius = np.sqrt(dx**2 + dy**2)
    max_radius = np.sqrt(cx**2 + cy**2) * 1.5 + 1e-6

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.maximum(0.0, 1.0 - radius / max_radius)
        pinch_factor = 1.0 - power * (factor ** 2)
        
        # 歪み適用
        new_x = cx + dx / pinch_factor
        new_y = cy + dy / pinch_factor
        
        # pinch_factorが小さすぎる（中心付近）場合は処理しない(外へ飛ばす)
        mask = pinch_factor <= 0.01
        new_x[mask] = cx + dx[mask] * 100
        new_y[mask] = cy + dy[mask] * 100

    map_x = new_x
    map_y = new_y

    distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return distorted, map_x, map_y

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    """YOLOフォーマットで保存"""
    xmin, ymin, xmax, ymax = bbox_coords
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    bw = xmax - xmin
    bh = ymax - ymin
    
    # 幅や高さが0以下の場合は保存しない
    if bw <= 0 or bh <= 0:
        return

    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0

    norm_xc = xc / img_w
    norm_yc = yc / img_h
    norm_w = bw / img_w
    norm_h = bh / img_h

    line = f"0 {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}\n"
    
    with open(filepath, "w") as f:
        f.write(line)

def process_images():
    # FaceMeshのみを使用
    mp_face_mesh = mp.solutions.face_mesh
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    
    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return

    effect_names = ['melt', 'squeeze']

    # FaceMeshの設定
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,       # 静止画モード
        max_num_faces=5,              # 最大検出数
        refine_landmarks=True,        # 精度の高いランドマークを使用
        min_detection_confidence=0.5  # 信頼度閾値
    ) as face_mesh:
        
        for filename in files:
            input_path = os.path.join(INPUT_DIR, filename)
            file_bytes = np.fromfile(input_path, np.uint8)
            img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_original is None: continue
            
            print(f"--- 処理開始: {filename} ---")
            h_img, w_img, _ = img_original.shape
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            
            # FaceMeshのみを実行
            results = face_mesh.process(img_rgb)

            # FaceMeshで検出できなかった場合は即座にスキップ
            if not results.multi_face_landmarks:
                print(f"スキップ: {filename} (FaceMeshで検出不可)")
                continue

            # 検出できた場合のみ処理続行
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # 4点のランドマーク座標を取得
                pts = {}
                for key, idx in LM_INDICES.items():
                    lm = face_landmarks.landmark[idx]
                    pts[key] = (int(lm.x * w_img), int(lm.y * h_img))
                
                # 顔全体の矩形を算出（ROI用）
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                face_bbox_xmin = int(min(xs) * w_img)
                face_bbox_ymin = int(min(ys) * h_img)
                face_bbox_xmax = int(max(xs) * w_img)
                face_bbox_ymax = int(max(ys) * h_img)
                face_w = face_bbox_xmax - face_bbox_xmin
                face_h = face_bbox_ymax - face_bbox_ymin

                if face_w <= 0 or face_h <= 0: continue

                for effect_name in effect_names:
                    output_img = img_original.copy()
                    
                    # ランダムパラメータ
                    rnd_squeeze_power = random.uniform(0.4, 0.95)
                    rnd_melt_power = random.uniform(0.4, 0.95)
                    val1, val2 = random.uniform(0.4, 0.95), random.uniform(0.4, 0.95)
                    rnd_start_rel = min(val1, val2)
                    rnd_level_rel = max(val1, val2)
                    if rnd_level_rel - rnd_start_rel < 0.1:
                        if rnd_level_rel < 0.85: rnd_level_rel += 0.1
                        else: rnd_start_rel -= 0.1

                    # ROI計算（下方向を広く取る）
                    margin_x = 0.1
                    margin_top = 0.1
                    margin_bottom = 2.5
                    
                    x_m = int(face_bbox_xmin - face_w * margin_x)
                    y_m = int(face_bbox_ymin - face_h * margin_top)
                    w_m = int(face_w * (1 + margin_x * 2))
                    h_m = int(face_h * (1 + margin_top + margin_bottom))
                    
                    x1, y1 = max(0, x_m), max(0, y_m)
                    x2, y2 = min(w_img, x_m + w_m), min(h_img, y_m + h_m)
                    
                    roi = img_original[y1:y2, x1:x2]
                    if roi.shape[0] == 0 or roi.shape[1] == 0: continue
                    
                    # アゴの相対位置
                    face_chin_y = pts['bottom'][1] - y1
                    
                    # エフェクト適用
                    if effect_name == 'melt':
                        glitched_roi, map_x, map_y = apply_vertical_melt(roi, face_chin_y, rnd_melt_power, rnd_start_rel, rnd_level_rel)
                    else:
                        glitched_roi, map_x, map_y = apply_squeeze_pinch(roi, face_chin_y, rnd_squeeze_power)

                    # --- 座標追跡 ---
                    roi_offset = (x1, y1)
                    x_coords = []
                    y_coords = []

                    for key, pt in pts.items():
                        # ROI内に点が含まれる場合のみ移動先を計算
                        if x1 <= pt[0] < x2 and y1 <= pt[1] < y2:
                            new_pt = get_new_coords(pt, map_x, map_y, roi_offset)
                            x_coords.append(new_pt[0])
                            y_coords.append(new_pt[1])
                        else:
                            # 範囲外なら元の座標を使用
                            x_coords.append(pt[0])
                            y_coords.append(pt[1])

                    # 画像合成
                    output_img[y1:y2, x1:x2] = glitched_roi
                    
                    # バウンディングボックス計算
                    if x_coords and y_coords:
                        new_xmin = min(x_coords)
                        new_ymin = min(y_coords)
                        new_xmax = max(x_coords)
                        new_ymax = max(y_coords)
                        
                        # 保存
                        name, ext = os.path.splitext(filename)
                        base_name = f"{name}_{effect_name}_{i}"
                        img_save_path = os.path.join(OUTPUT_DIR, f"{base_name}{ext}")
                        txt_save_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

                        is_success, buffer = cv2.imencode(ext, output_img)
                        if is_success:
                            with open(img_save_path, "wb") as f: f.write(buffer)
                            
                        save_yolo_format(txt_save_path, w_img, h_img, (new_xmin, new_ymin, new_xmax, new_ymax))
                        print(f"    -> 保存完了: {base_name}")

    print("\nすべての処理が完了しました。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()