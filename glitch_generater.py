import cv2
import mediapipe as mp
import numpy as np
import torch
import os  # フォルダ操作用に追加
from iopaint.model.lama import LaMa
from iopaint.schema import InpaintRequest, HDStrategy

# --- 設定 ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. モデルのロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaMa(device)

# 目のインデックス
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def get_masks(img):
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

        if res_det.detections and res_mesh.multi_face_landmarks:
            for face_landmarks, detection in zip(res_mesh.multi_face_landmarks, res_det.detections):
                bbox = detection.location_data.relative_bounding_box
                
                fx = int(bbox.xmin * w)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                y_cut = int(bbox.ymin * h) + fh + int(fh * 0.1)

                x_start = max(0, fx - int(fw * 0.8))
                x_end = min(w - 1, fx + fw + int(fw * 0.8))

                temp_head = np.zeros((h, w), dtype=np.uint8)
                temp_head[:y_cut, x_start:x_end] = person_mask[:y_cut, x_start:x_end]
                
                kernel = np.ones((7, 7), np.uint8)
                temp_head = cv2.dilate(temp_head, kernel, iterations=2)
                cv2.circle(temp_head, (int((bbox.xmin + bbox.width/2)*w), y_cut), int(fh*0.2), 255, -1)
                temp_head[y_cut:, :] = 0
                
                full_head_mask = cv2.bitwise_or(full_head_mask, temp_head)

                # --- 修正: 黒目（虹彩）を中心に正円でくりぬく処理 ---
                # 虹彩の中心: 左目(468), 右目(473)
                # 直径計算用の端点: 左目(33, 133), 右目(362, 263)
                eye_configs = [
                    {'iris': 468, 'corners': (33, 133)}, # 左目
                    {'iris': 473, 'corners': (362, 263)} # 右目
                ]
                
                for config in eye_configs:
                    # 黒目（虹彩）の中心座標を取得
                    iris_lm = face_landmarks.landmark[config['iris']]
                    cx, cy = int(iris_lm.x * w), int(iris_lm.y * h)
                    
                    # 目の端点の距離から半径を計算
                    p1 = face_landmarks.landmark[config['corners'][0]]
                    p2 = face_landmarks.landmark[config['corners'][1]]
                    
                    # ユークリッド距離で左右幅を算出
                    # $\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
                    dist = np.sqrt(((p2.x - p1.x) * w)**2 + ((p2.y - p1.y) * h)**2)
                    radius = int(dist / 2)
                    
                    # 黒目を中心に正円を描画
                    cv2.circle(eye_only_mask, (cx, cy), radius, 255, -1)

            return full_head_mask, eye_only_mask, True
    return None, None, False

# --- メイン処理ループ ---
image_extensions = (".png", ".jpg", ".jpeg", ".webp")
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

if not files:
    print(f"'{INPUT_DIR}' フォルダに画像を入れてから実行してください。")

for filename in files:
    input_path = os.path.join(INPUT_DIR, filename)
    
    # 修正: 日本語パスにも対応した堅牢な読み込み方法
    # cv2.imread(input_path) の代わりにこちらを使用
    file_bytes = np.fromfile(input_path, np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # ガード処理: 読み込みに失敗した場合はスキップ
    if img_original is None:
        print(f"警告: {filename} を画像として読み込めませんでした。スキップします。")
        continue
    
    print(f"処理中: {filename}...")
    head_mask, eye_mask, found = get_masks(img_original)

    if found:
        # 2. LaMa で消去
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        config = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)
        res_lama = model(img_rgb, head_mask, config)
        
        # 型変換と正規化
        if res_lama.max() <= 1.0: res_lama = res_lama * 255
        bg_img = np.clip(res_lama, 0, 255).astype(np.uint8)

        # 3. 高画質を維持する合成（顔なし）
        keep_body_mask = cv2.bitwise_not(head_mask)
        headless_img = bg_img.copy()
        headless_img[keep_body_mask == 255] = img_original[keep_body_mask == 255]
        
        # 保存
        name, ext = os.path.splitext(filename)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_headless{ext}"), headless_img)

        # 4. 目の部分だけを上書き（目あり）
        floating_eyes_img = headless_img.copy()
        floating_eyes_img[eye_mask == 255] = img_original[eye_mask == 255]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_eyes{ext}"), floating_eyes_img)

        print(f"完了: {filename}")
    else:
        print(f"スキップ: {filename} (顔が検出されませんでした)")

print("\nすべての処理が終了しました。outputフォルダを確認してください。")