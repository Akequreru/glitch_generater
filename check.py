import cv2
import os
import glob
import random

# --- 設定 ---
IMAGE_DIR = 'output_2/image'
LABEL_DIR = 'output_2/label'
CLASSES_FILE = 'classes.txt'
WINDOW_NAME = "YOLO Annotation Verification"

MAX_WIDTH = 1280
MAX_HEIGHT = 720

CLASS_COLORS = {}


def get_unique_color(class_name):
    if class_name not in CLASS_COLORS:
        CLASS_COLORS[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return CLASS_COLORS[class_name]

def load_classes(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        return {i: name for i, name in enumerate(class_list)}
    except FileNotFoundError:
        print(f"❌ エラー: '{file_path}' が見つかりません。")
        return {}

def get_yolo_annotations(label_path, img_w, img_h, class_map):
    annotations = []
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            class_id = int(parts[0])
            x_c, y_c, w_norm, h_norm = [float(p) for p in parts[1:]]
            x_min = int((x_c - w_norm / 2) * img_w)
            y_min = int((y_c - h_norm / 2) * img_h)
            x_max = int((x_c + w_norm / 2) * img_w)
            y_max = int((y_c + h_norm / 2) * img_h)
            annotations.append({'bbox': [x_min, y_min, x_max, y_max], 'label': class_map.get(class_id, f"ID:{class_id}")})
    return annotations

def check_mismatch():
    """画像とラベルのペアをチェックして不足を羅列する"""
    print(f"\n{'='*20} 整合性チェック開始 {'='*20}")
    
    # 全画像ファイルのベース名を取得
    img_files = {}
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for p in glob.glob(os.path.join(IMAGE_DIR, ext)):
            base = os.path.splitext(os.path.basename(p))[0]
            img_files[base] = os.path.basename(p)
            
    # 全ラベルファイルのベース名を取得
    label_files = {}
    for p in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
        base = os.path.splitext(os.path.basename(p))[0]
        label_files[base] = os.path.basename(p)
        
    img_bases = set(img_files.keys())
    label_bases = set(label_files.keys())
    
    # 1. 画像はあるがラベルがない
    missing_labels = sorted(list(img_bases - label_bases))
    # 2. ラベルはあるが画像がない
    missing_images = sorted(list(label_bases - img_bases))
    
    if not missing_labels and not missing_images:
        print("✅ すべてのファイルが正しくペアになっています！")
    else:
        if missing_labels:
            print(f"\n⚠️ 【ラベル不足】画像はあるがtxtがない ({len(missing_labels)}件):")
            for b in missing_labels:
                print(f"  - {img_files[b]}")
        
        if missing_images:
            print(f"\n⚠️ 【画像不足】txtはあるが画像がないため、削除します ({len(missing_images)}件):")
            for b in missing_images:
                # --- 修正箇所 ---
                # 正しいパスを構築して削除を実行
                target_label_path = os.path.join(LABEL_DIR, label_files[b])
                try:
                    os.remove(target_label_path)
                    print(f"  [Deleted] - {label_files[b]}")
                except Exception as e:
                    print(f"  ❌ 削除失敗: {label_files[b]} ({e})")
    print(f"\n{'='*55}")

def visualize_yolo():
    class_map = load_classes(CLASSES_FILE)
    image_paths = sorted([p for ext in ['*.jpg', '*.jpeg', '*.png'] for p in glob.glob(os.path.join(IMAGE_DIR, ext))])
    
    if not image_paths:
        print(f"❌ エラー: 画像が {IMAGE_DIR} に見つかりません。")
        return

    print("\n--- 可視化モード開始 ---")
    print("D or Any: 次へ / A: 前へ / Q: 終了")

    idx = 0
    num_images = len(image_paths)

    while 0 <= idx < num_images:
        img_path = image_paths[idx]
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]
        label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue
        
        h, w, _ = img.shape
        annos = get_yolo_annotations(label_path, w, h, class_map)
        temp_img = img.copy()

        info_text = f"[{idx + 1} / {num_images}] {img_filename}"
        cv2.putText(temp_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        for ann in annos:
            x1, y1, x2, y2 = ann['bbox']
            label = ann['label']
            color = get_unique_color(label)
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 > 25 else y1 + 25
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(temp_img, (x1, text_y - th - 5), (x1 + tw, text_y + 5), color, -1)
            cv2.putText(temp_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if w > MAX_WIDTH or h > MAX_HEIGHT:
            scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
            display_img = cv2.resize(temp_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display_img = temp_img

        cv2.imshow(WINDOW_NAME, display_img)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            idx = max(0, idx - 1)
        else:
            idx += 1
            if idx >= num_images:
                print("すべての画像の確認が終了しました。")
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("実行モードを選択してください:")
    print("1: 可視化モード (画像を一枚ずつ確認)")
    print("2: 整合性チェックモード (不足ファイルをリストアップ)")
    
    choice = input("選択 (1 or 2): ")
    
    if choice == '1':
        visualize_yolo()
    elif choice == choice == '2':
        check_mismatch()
    else:
        print("無効な選択です。終了します。")