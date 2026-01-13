import cv2
import mediapipe as mp
import numpy as np
import os

# --- 設定 ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image extension settings (added .bmp support)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# MediaPipe Face Meshのランドマークインデックス
LM_INDICES = {
    # 左目（内側）
    'left_eye_contour': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    # 右目（内側）
    'right_eye_contour': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    # 口の外周輪郭
    'lips_contour': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17]
}

def expand_contour(contour, scale=1.8):
    """
    輪郭を中心から外側に拡大する関数
    scale: 拡大倍率 (1.5〜2.0程度で周囲の皮膚を含める)
    """
    if len(contour) == 0: return contour
    M = cv2.moments(contour)
    if M['m00'] == 0: return contour
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cnt_norm = contour - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_new = cnt_scaled + [cx, cy]
    return cnt_new.astype(np.int32)

def apply_omni_directional_stretch(roi, contour_points, roi_offset):
    """
    指定された輪郭に向かって周囲から皮膚を引き延ばす
    """
    h, w, _ = roi.shape
    off_x, off_y = roi_offset

    # 1. 輪郭をROI相対座標に変換し、拡大する
    rel_contour = np.array([(p[0] - off_x, p[1] - off_y) for p in contour_points], dtype=np.int32)
    # scaleを調整することで隠す範囲の広さを変えられます（口と目で変えても良い）
    scale_factor = 1.8 
    rel_contour = expand_contour(rel_contour, scale=scale_factor)

    # 2. 中心とサイズ計算
    M = cv2.moments(rel_contour)
    if M['m00'] == 0: return roi
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    _, _, w_cnt, h_cnt = cv2.boundingRect(rel_contour)
    
    # 3. マップ作成
    grid_x, grid_y = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))
    vec_x, vec_y = grid_x - cx, grid_y - cy
    distance = np.sqrt(vec_x**2 + vec_y**2) + 1e-6
    unit_x, unit_y = vec_x / distance, vec_y / distance

    # 4. 変形マップ生成（ストレッチ強度調整）
    # 輪郭の大きさに比例して引き延ばす距離を決める
    stretch_dist_x = w_cnt * 0.6
    stretch_dist_y = h_cnt * 0.6
    map_x = grid_x + unit_x * stretch_dist_x
    map_y = grid_y + unit_y * stretch_dist_y

    # 5. マスク作成とリマップ
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [rel_contour], 1.0)
    mask = cv2.GaussianBlur(mask, (21, 21), 0) # 境界をぼかす
    mask_3ch = np.dstack([mask]*3)
    
    warped_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    final_roi = warped_roi * mask_3ch + roi.astype(np.float32) * (1.0 - mask_3ch)
    
    return final_roi.astype(np.uint8)

def save_yolo_format(filepath, img_w, img_h, bbox_coords):
    """YOLOフォーマット保存"""
    xmin, ymin, xmax, ymax = bbox_coords
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0: return
    xc, yc = (xmin + bw / 2.0) / img_w, (ymin + bh / 2.0) / img_h
    bw, bh = bw / img_w, bh / img_h
    # 範囲外に出ないようにクリップ
    xc, yc = np.clip([xc, yc], 0.0, 1.0)
    bw, bh = np.clip([bw, bh], 0.0, 1.0)
    
    with open(filepath, "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

def calculate_roi_and_bbox(img_shape, target_points):
    """対象点群からROIとYOLO用BBoxを計算するヘルパー関数"""
    h_img, w_img = img_shape[:2]
    xs = [p[0] for p in target_points]
    ys = [p[1] for p in target_points]
    
    # 対象領域の最小矩形
    target_xmin, target_ymin = min(xs), min(ys)
    target_xmax, target_ymax = max(xs), max(ys)
    target_w, target_h = target_xmax - target_xmin, target_ymax - target_ymin
    
    # ROI計算（対象領域より広めに取る）
    margin = 1.0 # マージン率
    roi_cx, roi_cy = (target_xmin + target_xmax) // 2, (target_ymin + target_ymax) // 2
    roi_size = max(target_w, target_h) * (1 + margin)
    
    x1 = max(0, int(roi_cx - roi_size / 2))
    y1 = max(0, int(roi_cy - roi_size / 2))
    x2 = min(w_img, int(roi_cx + roi_size / 2))
    y2 = min(h_img, int(roi_cy + roi_size / 2))
    
    roi_src = img_original[y1:y2, x1:x2] # グローバル変数を参照せず引数で渡すのが理想だが簡便のため
    if roi_src.shape[0] == 0 or roi_src.shape[1] == 0: return None

    # ROI座標(x1,y1,x2,y2), YOLO用BBox(xmin,ymin,xmax,ymax), ROI画像 を返す
    return (x1, y1, x2, y2), (target_xmin, target_ymin, target_xmax, target_ymax), roi_src

def process_and_save(img_original, filename, face_idx, suffix, target_contours_keys, pts):
    """指定された部位を処理して保存する共通関数"""
    h_img, w_img = img_original.shape[:2]
    
    # 対象となるすべての点を集める
    all_target_pts = []
    for key in target_contours_keys:
        all_target_pts.extend(pts[key])
        
    # ROIとBBoxの計算
    roi_data = calculate_roi_and_bbox(img_original.shape, all_target_pts)
    if roi_data is None: return

    (x1, y1, x2, y2), bbox_coords, roi_src = roi_data
    roi_offset = (x1, y1)
    
    # ストレッチ処理の適用
    roi_working = roi_src.copy()
    for key in target_contours_keys:
        roi_working = apply_omni_directional_stretch(roi_working, pts[key], roi_offset)
        
    # 元画像に合成して保存
    output_img = img_original.copy()
    output_img[y1:y2, x1:x2] = roi_working
    
    name, ext = os.path.splitext(filename)
    base_name = f"{name}_face{face_idx}_{suffix}"
    img_save_path = os.path.join(OUTPUT_DIR, f"{base_name}{ext}")
    txt_save_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    
    # 画像保存 (日本語ファイル名対応のためimencodeを使用)
    is_success, buffer = cv2.imencode(ext, output_img)
    if is_success:
        with open(img_save_path, "wb") as f: f.write(buffer)
        # YOLOフォーマットtxt保存
        save_yolo_format(txt_save_path, w_img, h_img, bbox_coords)
        print(f"    -> 保存完了 [{suffix}]: {base_name}{ext}")


def process_images():
    global img_original # calculate_roi_and_bboxから参照するため
    mp_face_mesh = mp.solutions.face_mesh
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if not files:
        print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")
        return

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        
        for filename in files:
            input_path = os.path.join(INPUT_DIR, filename)
            # 日本語ファイル名対応のためnp.fromfileを使用
            try:
                file_bytes = np.fromfile(input_path, np.uint8)
                img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_original is None: raise Exception("Decode failed")
            except Exception as e:
                print(f"警告: ファイル {filename} を読み込めませんでした。スキップします。")
                continue
            
            print(f"--- 処理開始: {filename} ---")
            h_img, w_img, _ = img_original.shape
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                print("  顔が検出されませんでした。")
                continue

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # 1. 全ランドマーク座標の取得
                pts = {}
                for key, indices in LM_INDICES.items():
                    pts[key] = [(int(face_landmarks.landmark[idx].x * w_img), int(face_landmarks.landmark[idx].y * h_img)) for idx in indices]

                # 2. 【目】の処理と保存 (目が隠れた画像が生成される)
                process_and_save(
                    img_original, filename, i, "eyes",
                    ['left_eye_contour', 'right_eye_contour'], pts
                )

                # 3. 【口】の処理と保存 (口が隠れた画像が別途生成される)
                process_and_save(
                    img_original, filename, i, "mouth",
                    ['lips_contour'], pts
                )

    print("\nすべての処理が完了しました。outputフォルダを確認してください。")

if __name__ == "__main__":
    process_images()