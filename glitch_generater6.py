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

# MediaPipe Face Meshのランドマークインデックス
# 4点だけではなく、顔の輪郭全体（Face Oval）を使用します
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

def get_new_coords(src_pt, map_x, map_y, roi_offset):
    """
    変形後の座標をマップから逆算する
    （元画像の座標 src_pt が、変形後の画像のどこに移動したかを探す）
    """
    src_x, src_y = src_pt
    off_x, off_y = roi_offset
    roi_src_x = src_x - off_x
    roi_src_y = src_y - off_y

    h, w = map_x.shape
    # 範囲外チェック
    if not (0 <= roi_src_x < w and 0 <= roi_src_y < h):
        return src_pt

    # マップ上で最も近いピクセルを探す（逆写像の反転）
    dist_map = (map_x - roi_src_x)**2 + (map_y - roi_src_y)**2
    min_idx = np.argmin(dist_map)
    dst_y_rel, dst_x_rel = np.unravel_index(min_idx, dist_map.shape)

    dst_x = dst_x_rel + off_x
    dst_y = dst_y_rel + off_y
    return (int(dst_x), int(dst_y))

def apply_random_suction(roi, target_point, face_size, power):
    """
    指定したランダムなターゲット点に向かって縮む（吸い込まれる）エフェクト
    """
    h, w, _ = roi.shape
    cx, cy = target_point # ランダムなターゲットを中心にする
    
    # グリッド計算
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    radius = np.sqrt(dx**2 + dy**2)
    
    # 影響範囲（顔のサイズの2.0倍程度まで広げる）
    max_radius = face_size * 2.0 + 1e-6

    with np.errstate(divide='ignore', invalid='ignore'):
        # 中心に近いほど factor が 1.0 に近づく
        factor = np.maximum(0.0, 1.0 - radius / max_radius)
        
        # 歪み係数計算
        # power乗することで吸い込みカーブを調整
        pinch_factor = 1.0 - power * (factor ** 2)
        
        # 逆写像座標の計算
        # pinch_factor < 1.0 なので、参照先が中心から遠ざかる ＝ 画像は中心に集まる
        new_x = cx + dx / pinch_factor
        new_y = cy + dy / pinch_factor
        
        # 特異点処理（過剰な値を防ぐ）
        mask = pinch_factor <= 0.01
        new_x[mask] = cx + dx[mask] * 100
        new_y[mask] = cy + dy[mask] * 100

    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)

    # 変形実行
    distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return distorted, map_x, map_y

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    xmin, ymin, xmax, ymax = bbox_coords
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(img_w, xmax); ymax = min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    
    if bw <= 0 or bh <= 0: return # 無効なBBoxは保存しない

    xc, yc = xmin + bw / 2.0, ymin + bh / 2.0
    with open(filepath, "w") as f:
        f.write(f"0 {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

def process_images():
    mp_face_mesh = mp.solutions.face_mesh
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    
    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return

    # エフェクト設定
    effect_names = ['squeeze']

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        
        for filename in files:
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
                # --- 顔の輪郭（Face Oval）全点の座標を取得 ---
                pts_oval = []
                all_xs, all_ys = [], []
                
                for idx in FACE_OVAL_INDICES:
                    lm = face_landmarks.landmark[idx]
                    px, py = int(lm.x * w_img), int(lm.y * h_img)
                    pts_oval.append((px, py))
                    all_xs.append(px)
                    all_ys.append(py)
                
                # 顔サイズと中心計算（すべての輪郭点から算出）
                face_w = max(all_xs) - min(all_xs)
                face_h = max(all_ys) - min(all_ys)
                face_size = max(face_w, face_h)
                face_cx = int(sum(all_xs) / len(all_xs))
                face_cy = int(sum(all_ys) / len(all_ys))
                
                # ROI計算（歪みで移動するため広めに取る）
                margin = 0.8
                roi_len = int(face_size * (1 + margin))
                
                x1 = max(0, face_cx - roi_len)
                y1 = max(0, face_cy - roi_len)
                x2 = min(w_img, face_cx + roi_len)
                y2 = min(h_img, face_cy + roi_len)
                
                roi = img_original[y1:y2, x1:x2]
                if roi.shape[0] == 0 or roi.shape[1] == 0: continue
                roi_h, roi_w = roi.shape[:2]

                for effect_name in effect_names:
                    output_img = img_original.copy()
                    
                    # --- ランダムターゲット決定 ---
                    angle_rnd = random.uniform(0, 2 * math.pi)
                    # 顔の中心から顔の半径の0〜0.8倍の距離（顔内〜顔周辺）
                    dist_rnd = face_size * random.uniform(0.0, 0.8)
                    
                    target_abs_x = face_cx + dist_rnd * math.cos(angle_rnd)
                    target_abs_y = face_cy + dist_rnd * math.sin(angle_rnd)
                    
                    # ROI内相対座標
                    target_rel_x = target_abs_x - x1
                    target_rel_y = target_abs_y - y1
                    target_pt = (target_rel_x, target_rel_y)

                    # パワー設定
                    rnd_power = random.uniform(0.6, 0.95)

                    # エフェクト適用
                    glitched_roi, map_x, map_y = apply_random_suction(
                        roi, target_pt, face_size, rnd_power)

                    # --- 座標追跡（修正点） ---
                    # 4点ではなく、顔の輪郭全点（36点）を追跡してBBoxを計算
                    roi_offset = (x1, y1)
                    x_coords, y_coords = [], []
                    
                    for pt in pts_oval:
                        if x1 <= pt[0] < x2 and y1 <= pt[1] < y2:
                            new_pt = get_new_coords(pt, map_x, map_y, roi_offset)
                            x_coords.append(new_pt[0])
                            y_coords.append(new_pt[1])
                        else:
                            # 範囲外の場合は元の座標を使う
                            x_coords.append(pt[0])
                            y_coords.append(pt[1])

                    # 画像書き戻し
                    output_img[y1:y2, x1:x2] = glitched_roi
                    
                    # BBox保存
                    if x_coords:
                        new_xmin, new_ymin = min(x_coords), min(y_coords)
                        new_xmax, new_ymax = max(x_coords), max(y_coords)
                        
                        # アノテーション領域に少しマージンを持たせる（オプション）
                        # ギリギリすぎると検出精度が落ちる場合があるため
                        bbox_margin = 0 # 必要に応じて数ピクセル足してください
                        new_xmin -= bbox_margin; new_ymin -= bbox_margin
                        new_xmax += bbox_margin; new_ymax += bbox_margin
                        
                        new_bbox = (new_xmin, new_ymin, new_xmax, new_ymax)
                        
                        save_base = f"{os.path.splitext(filename)[0]}_{effect_name}_{i}"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{save_base}.jpg"), output_img)
                        save_yolo_format(os.path.join(OUTPUT_DIR, f"{save_base}.txt"), w_img, h_img, new_bbox)
                        print(f"    -> 保存: {save_base}")

    print("\nすべての処理が完了しました。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()