import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

img = cv2.imread('person.png')
if img is None:
    print("Error: person.png が読み込めません。")
    exit()

h, w, _ = img.shape
result = img.copy()

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 顔全体の範囲を把握するために全ランドマークの最小/最大座標を取得
            all_x = [lm.x * w for lm in face_landmarks.landmark]
            all_y = [lm.y * h for lm in face_landmarks.landmark]
            
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))
            
            face_w = x_max - x_min
            face_h = y_max - y_min
            
            # --- マスク範囲の強制拡張 (耳と頭頂部を飲み込む) ---
            # 左右に顔幅の30%ずつ広げる (これで耳をカバー)
            x_start = max(0, x_min - int(face_w * 0.3))
            x_end = min(w - 1, x_max + int(face_w * 0.3))
            
            # 上に顔高の80%広げる (これで髪の毛と頭頂部をカバー)
            y_start = max(0, y_min - int(face_h * 0.8))
            
            # 下（カットライン）: 顎の少し下、首の付け根あたりでスパッと切る
            # ここを y_max + (高さの10%) 程度に固定
            y_cut = min(h - 1, y_max + int(face_h * 0.1))

            # --- 背景引き伸ばしによる塗りつぶし ---
            for y_line in range(y_start, y_cut + 1):
                # マスク範囲の「さらに外側」から背景色を取得
                # 左右に10ピクセルほど余裕を持たせた地点の色を使う
                bg_x_left = max(0, x_start - 10)
                bg_x_right = min(w - 1, x_end + 10)
                
                color_left = img[y_line, bg_x_left]
                color_right = img[y_line, bg_x_right]
                
                # 指定範囲を横方向に塗りつぶす
                for x_pos in range(x_start, x_end + 1):
                    # 左右の背景色を滑らかにブレンド
                    weight = (x_pos - x_start) / (x_end - x_start + 1)
                    mixed_color = color_left * (1 - weight) + color_right * weight
                    result[y_line, x_pos] = mixed_color.astype(np.uint8)

        cv2.imwrite('headless_person.png', result)
        print("成功: 耳と頭部を完全に消去し、背景で上書きしました。")
    else:
        print("顔が検出されませんでした。")

        