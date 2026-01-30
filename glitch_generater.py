import cv2
import mediapipe as mp
import numpy as np
import torch
import os
import random
from iopaint.model.lama import LaMa
from iopaint.schema import InpaintRequest, HDStrategy

# =========================================================
# 1. 設定 & 定数
# =========================================================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOREHEAD_EXTEND_RATIO = 0.07  # 移植: 額の拡張比率

# 1. モデルのロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaMa(device)

# =========================================================
# 2. アノテーション・ヘルパー関数 (移植)
# =========================================================
def save_yolo_format(filepath, img_w, img_h, bbox_coords, class_id):
    """
    xmin, ymin, xmax, ymax を受け取り、指定された class_id で保存
    """
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
        f.write(f"{class_id} {n_xc:.6f} {n_yc:.6f} {n_bw:.6f} {n_bh:.6f}\n")

# =========================================================
# 3. マスク生成 & 座標取得ロジック
# =========================================================
def get_masks_and_coords(img):
    h, w, _ = img.shape
    mp_selfie = mp.solutions.selfie_segmentation
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_det = mp.solutions.face_detection

    with mp_selfie.SelfieSegmentation(model_selection=1) as selfie_seg, \
         mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, \
         mp_face_det.FaceDetection(model_selection=1) as face_det:
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_seg = selfie_seg.process(img_rgb)
        res_mesh = face_mesh.process(img_rgb)
        res_det = face_det.process(img_rgb)
        
        person_mask = (res_seg.segmentation_mask > 0.5).astype(np.uint8) * 255
        full_head_mask = np.zeros((h, w), dtype=np.uint8)
        eye_only_mask = np.zeros((h, w), dtype=np.uint8)
        
        # アノテーション用の座標リスト
        all_face_bboxes = []

        if res_det.detections and res_mesh.multi_face_landmarks:
            for face_landmarks, detection in zip(res_mesh.multi_face_landmarks, res_det.detections):
                # --- 移植: アノテーション用座標計算 (Gen 7/8/9のロジック) ---
                xs_all = [lm.x for lm in face_landmarks.landmark]
                ys_all = [lm.y for lm in face_landmarks.landmark]
                
                f_xmin, f_xmax = int(min(xs_all) * w), int(max(xs_all) * w)
                raw_ymin, raw_ymax = int(min(ys_all) * h), int(max(ys_all) * h)
                f_h_raw = raw_ymax - raw_ymin
                
                # 額の拡張を反映
                margin_up = int(f_h_raw * FOREHEAD_EXTEND_RATIO)
                f_ymin = max(0, raw_ymin - margin_up)
                f_ymax = raw_ymax
                
                all_face_bboxes.append((f_xmin, f_ymin, f_xmax, f_ymax))

                # --- LaMa用マスク作成ロジック ---
                bbox = detection.location_data.relative_bounding_box
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                y_cut = int(bbox.ymin * h) + fh + int(fh * 0.1)
                x_start = max(0, int(bbox.xmin * w) - int(fw * 0.8))
                x_end = min(w - 1, int(bbox.xmin * w) + fw + int(fw * 0.8))

                temp_head = np.zeros((h, w), dtype=np.uint8)
                temp_head[:y_cut, x_start:x_end] = person_mask[:y_cut, x_start:x_end]
                
                kernel = np.ones((7, 7), np.uint8)
                temp_head = cv2.dilate(temp_head, kernel, iterations=2)
                cv2.circle(temp_head, (int((bbox.xmin + bbox.width/2)*w), y_cut), int(fh*0.2), 255, -1)
                temp_head[y_cut:, :] = 0
                
                full_head_mask = cv2.bitwise_or(full_head_mask, temp_head)

                # 目（虹彩）のマスク
                eye_configs = [{'iris': 468, 'corners': (33, 133)}, {'iris': 473, 'corners': (362, 263)}]
                for config in eye_configs:
                    iris_lm = face_landmarks.landmark[config['iris']]
                    cx, cy = int(iris_lm.x * w), int(iris_lm.y * h)
                    p1, p2 = face_landmarks.landmark[config['corners'][0]], face_landmarks.landmark[config['corners'][1]]
                    dist = np.sqrt(((p2.x - p1.x) * w)**2 + ((p2.y - p1.y) * h)**2)
                    cv2.circle(eye_only_mask, (cx, cy), int(dist / 2), 255, -1)

            return full_head_mask, eye_only_mask, all_face_bboxes, True
    return None, None, None, False

# =========================================================
# 4. メイン処理
# =========================================================
image_extensions = (".png", ".jpg", ".jpeg", ".webp")
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

for filename in files:
    input_path = os.path.join(INPUT_DIR, filename)
    file_bytes = np.fromfile(input_path, np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_original is None: continue
    
    print(f"処理中: {filename}...")
    h_img, w_img, _ = img_original.shape
    head_mask, eye_mask, bboxes, found = get_masks_and_coords(img_original)

    if found:
        # LaMa推論
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        res_lama = model(img_rgb, head_mask, InpaintRequest(hd_strategy=HDStrategy.ORIGINAL))
        if res_lama.max() <= 1.0: res_lama *= 255
        bg_img = np.clip(res_lama, 0, 255).astype(np.uint8)

        # 共通のベース画像（顔消し）
        headless_img = bg_img.copy()
        keep_body_mask = cv2.bitwise_not(head_mask)
        headless_img[keep_body_mask == 255] = img_original[keep_body_mask == 255]
        
        base_name, ext = os.path.splitext(filename)

        # --- 各パターンの保存 ---
        patterns = [
            ("headless", 2),
            ("eyes", 1),
            ("two_wide_rects", 1)
        ]

        for p_name, class_id in patterns:
            out_img = headless_img.copy()
            
            if p_name == "eyes":
                out_img[eye_mask == 255] = img_original[eye_mask == 255]
            
            elif p_name == "two_wide_rects":
                restore_mask = head_mask.copy()
                y_coords, x_coords = np.where(head_mask == 255)
                # 代表的なBBox（最初の顔）を基準にサイズ決定
                f_xmin, f_ymin, f_xmax, f_ymax = bboxes[0]
                fw, fh = f_xmax - f_xmin, f_ymax - f_ymin
                
                for _ in range(2):
                    idx = random.randint(0, len(x_coords) - 1)
                    rx, ry = x_coords[idx], y_coords[idx]
                    rw, rh = random.randint(int(fw*0.4), int(fw*0.7)), random.randint(int(fh*0.15), int(fh*0.3))
                    cv2.rectangle(restore_mask, (rx-rw//2, ry-rh//2), (rx+rw//2, ry+rh//2), 0, -1)
                
                restore_mask = cv2.bitwise_and(restore_mask, head_mask)
                out_img[restore_mask == 255] = img_original[restore_mask == 255]

            # 画像とアノテーションの保存
            sname = f"{base_name}_{p_name}"
            cv2.imencode(ext, out_img)[1].tofile(os.path.join(OUTPUT_DIR, sname + ext))
            
            # 移植した正確なBBoxでYOLO保存 (複数顔対応)
            # ※YOLOは1ファイルに複数行書けるため、appendモードで全顔分書き出し
            with open(os.path.join(OUTPUT_DIR, sname + ".txt"), "w") as f:
                for box in bboxes:
                    xmin, ymin, xmax, ymax = box
                    bw, bh = xmax - xmin, ymax - ymin
                    if bw <= 0 or bh <= 0: continue
                    xc, yc = xmin + bw/2.0, ymin + bh/2.0
                    f.write(f"{class_id} {xc/w_img:.6f} {yc/h_img:.6f} {bw/w_img:.6f} {bh/h_img:.6f}\n")

        print(f"完了: {filename}")
    else:
        print(f"スキップ: {filename}")