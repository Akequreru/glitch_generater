import cv2
import json
import os
import glob
import xml.etree.ElementTree as ET
import random

# --- 設定 ---
IMAGE_DIR = 'output'
CLASSES_FILE = 'classes.txt'
WINDOW_NAME = "Annotation Verification"

# 表示するウィンドウの最大サイズ（これに合わせて縮小します）
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 800

# フォルダ設定
OUTPUT_FOLDERS = {
    'yolo': 'output',
}

# クラスごとの色を保持する辞書
CLASS_COLORS = {}

def get_unique_color(class_name):
    """クラス名に基づいて一意の色を生成または取得する"""
    if class_name not in CLASS_COLORS:
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        CLASS_COLORS[class_name] = (b, g, r)
    return CLASS_COLORS[class_name]

def load_classes(file_path):
    """classes.txt を読み込み、IDと名前のマップを返す"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        
        for class_name in class_list:
            get_unique_color(class_name)
            
        return {i: name for i, name in enumerate(class_list)} 
    except FileNotFoundError:
        print(f"❌ エラー: クラスファイル '{file_path}' が見つかりません。")
        return {}

def get_annotations(img_filename_no_ext, img_filename, img_w, img_h, format_type):
    """指定された形式のアノテーションを読み込み、共通形式に変換する"""
    annotations = []
    class_map = load_classes(CLASSES_FILE)
    
    # --- YOLO 形式 ---
    if format_type == 'yolo':
        yolo_path = os.path.join(OUTPUT_FOLDERS['yolo'], f"{img_filename_no_ext}.txt")
        if not os.path.exists(yolo_path): return []
        
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                class_id = int(parts[0])
                x_c, y_c, w_norm, h_norm = [float(p) for p in parts[1:]]
                
                x = x_c * img_w; y = y_c * img_h
                w = w_norm * img_w; h = h_norm * img_h
                
                x_min = int(x - w / 2); y_min = int(y - h / 2)
                x_max = int(x + w / 2); y_max = int(y + h / 2)
                
                annotations.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'label': class_map.get(class_id, f"ID:{class_id}_Unknown")
                })
        
    # --- PASCAL VOC 形式 ---
    elif format_type == 'voc':
        voc_path = os.path.join(OUTPUT_FOLDERS['voc'], f"{img_filename_no_ext}.xml")
        if not os.path.exists(voc_path): return []

        try:
            tree = ET.parse(voc_path)
            root = tree.getroot()
        except ET.ParseError:
            print(f"⚠️ XMLパースエラー: {voc_path}")
            return []
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            x_min = int(float(bndbox.find('xmin').text))
            y_min = int(float(bndbox.find('ymin').text))
            x_max = int(float(bndbox.find('xmax').text))
            y_max = int(float(bndbox.find('ymax').text))
            
            annotations.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'label': label
            })
            
    # --- COCO 形式 ---
    elif format_type == 'coco': 
        coco_path = os.path.join(OUTPUT_FOLDERS['coco'], "dataset_coco.json") 
        if not os.path.exists(coco_path): return [] 

        with open(coco_path, 'r', encoding='utf-8') as f: 
            coco_data = json.load(f) 

        image_entry = None 
        for img in coco_data.get('images', []): 
            if img.get("file_name") == img_filename:
                image_entry = img 
                break 

        if image_entry is None: return [] 

        image_id = image_entry["id"] 
        coco_id_to_label = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])} 

        for ann in coco_data.get('annotations', []): 
            if ann['image_id'] != image_id: continue 

            x, y, w, h = ann['bbox']
            x_min = int(x) 
            y_min = int(y) 
            x_max = int(x + w) 
            y_max = int(y + h) 

            annotations.append({ 
                'bbox': [x_min, y_min, x_max, y_max], 
                'label': coco_id_to_label.get(ann['category_id'], f"ID:{ann['category_id']}_Unknown") 
            }) 

    return annotations

def visualize_annotations(format_type):
    """指定された形式の全てのアノテーションを画像上に描画し、検証する"""
    
    print(f"\n--- 検証開始: {format_type.upper()} 形式 ---")
    load_classes(CLASSES_FILE) 

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']: # 対応拡張子を少し増やしました
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_paths:
        print(f"❌ エラー: 画像ファイルが '{IMAGE_DIR}' に見つかりません。")
        return

    # ウィンドウをリサイズ可能にする設定
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        img_filename_no_ext, _ = os.path.splitext(img_filename)

        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape

        annotations = get_annotations(img_filename_no_ext, img_filename, img_w, img_h, format_type)

        if not annotations:
            print(f"⚠️ アノテーションなし: {img_filename}")
            # アノテーションがなくても画像は表示して確認できるようにcontinueしない方が親切な場合もありますが、
            # ここでは元のロジックに従いスキップ、あるいは枠なしで表示したい場合はここを調整してください
            # continue 

        temp_img = img.copy()
        
        # 描画
        for ann in annotations:
            x_min, y_min, x_max, y_max = ann['bbox']
            label = ann['label']
            color = get_unique_color(label) 
            
            cv2.rectangle(temp_img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(temp_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # --- 【修正箇所】画面サイズに合わせてリサイズ ---
        scale_w = MAX_DISPLAY_WIDTH / img_w
        scale_h = MAX_DISPLAY_HEIGHT / img_h
        scale = min(scale_w, scale_h)

        if scale < 0.7: # 画像が画面設定より大きい場合のみ縮小
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            display_img = cv2.resize(temp_img, (new_w, new_h))
        else:
            display_img = temp_img
        # ---------------------------------------------

        cv2.setWindowTitle(WINDOW_NAME, f"Verification ({format_type.upper()}): {img_filename} (Esc:Quit, Space/Enter:Next)")
        cv2.imshow(WINDOW_NAME, display_img)
        
        # キー入力待ち: 'q'か'Esc'で終了, それ以外で次へ
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27: # 27 is Esc
            break

    cv2.destroyAllWindows()
    print(f"検証 ({format_type.upper()}): 終了しました。")

if __name__ == "__main__":
    # 1. YOLO形式の検証
    visualize_annotations('yolo')