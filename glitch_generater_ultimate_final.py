import cv2
import mediapipe as mp
import numpy as np
import torch
import os
import random
import math
from iopaint.model.lama import LaMa
from iopaint.schema import InpaintRequest, HDStrategy

# =========================================================
# 1. 設定 & 定数 (参照スクリプトから完全移植)
# =========================================================
INPUT_DIR = "input_2"
OUTPUT_LABEL_DIR = "output_2/label"
OUTPUT_IMAGE_DIR = "output_2/image"
TEXTURE_DIR = "texture_input"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(TEXTURE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
FOREHEAD_EXTEND_RATIO = 0.07 


COLOR_THRESHOLDS = {
    'h_tol': 20, 's_tol_lower': 50, 's_tol_upper': 80,
    'v_tol_lower': 120, 'v_tol_upper': 100, 'cr_tol': 20, 'cb_tol': 20,
    'min_saturation_strict': 15, 'highlight_threshold': 90 
}

FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LM_INDICES = {
    'face_oval': FACE_OVAL_INDICES,
    'bottom': 152, 
    'left_eye_top_curve': [33, 246, 161, 160, 159, 158, 157, 173, 133],
    'left_eye_bottom_curve': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'right_eye_top_curve': [362, 398, 384, 385, 386, 387, 388, 466, 263],
    'right_eye_bottom_curve': [362, 382, 381, 380, 374, 373, 390, 249, 263],
    'left_eye_contour': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    'right_eye_contour': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    'lips_contour': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17],
    'left_eyebrow': [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    'right_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'lips_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
}
SKIN_SAMPLE_INDICES = [4, 234, 454, 152, 10]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # 強制的にCPUにする
model = LaMa(device)
print(f"使用デバイス: {device}")
if device.type == 'cuda':
    print(f"GPU名: {torch.cuda.get_device_name(0)}")

# =========================================================
# 2. 共通ヘルパー関数
# =========================================================

def save_yolo_format(filepath, img_w, img_h, bbox_coords, class_id):
    xmin, ymin, xmax, ymax = bbox_coords
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0: return
    xc, yc = xmin + bw / 2.0, ymin + bh / 2.0
    with open(filepath, "a") as f:
        f.write(f"{class_id} {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")

def get_roi_rect(center_x, center_y, width, height, margin, img_w, img_h):
    roi_size = max(width * (1 + margin), height * (1 + margin))
    x1, y1 = max(0, int(center_x - roi_size/2)), max(0, int(center_y - roi_size/2))
    x2, y2 = min(img_w, int(center_x + roi_size/2)), min(img_h, int(center_y + roi_size/2))
    return x1, y1, x2, y2

def get_new_coords(src_pt, map_x, map_y, roi_offset):
    src_x, src_y = src_pt
    off_x, off_y = roi_offset
    roi_src_x, roi_src_y = src_x - off_x, src_y - off_y
    h, w = map_x.shape
    if not (0 <= roi_src_x < w and 0 <= roi_src_y < h): return src_pt
    dist_map = (map_x - roi_src_x)**2 + (map_y - roi_src_y)**2
    min_idx = np.argmin(dist_map)
    dst_y_rel, dst_x_rel = np.unravel_index(min_idx, dist_map.shape)
    return (int(dst_x_rel + off_x), int(dst_y_rel + off_y))

def update_face_oval_points(face_oval_pts, transformation_maps):
    current_pts = list(face_oval_pts)
    for roi_offset, map_x, map_y in transformation_maps:
        if map_x is None or map_y is None: continue
        updated_pts = []
        off_x, off_y = roi_offset
        h, w = map_x.shape
        for pt in current_pts:
            if off_x <= pt[0] < off_x + w and off_y <= pt[1] < off_y + h:
                updated_pts.append(get_new_coords(pt, map_x, map_y, roi_offset))
            else: updated_pts.append(pt)
        current_pts = updated_pts
    return current_pts

def calculate_tight_bbox(points, img_w, img_h, extend_ratio=FOREHEAD_EXTEND_RATIO):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    margin_up = int((ymax - ymin) * extend_ratio)
    return (max(0, xmin), max(0, ymin - margin_up), min(img_w, xmax), min(img_h, ymax))

def extend_forehead_points(points, face_h, extension_ratio=0.15):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cy = sum(ys) / len(ys)
    min_y = min(ys)
    shift_amount = int(face_h * extension_ratio)
    modified_points = []
    for (x, y) in points:
        if y < cy and (y - min_y) < (face_h * 0.3):
            modified_points.append((x, max(0, y - shift_amount)))
        else:
            modified_points.append((x, y))
    return modified_points
# =========================================================
# 3. 幾何学変形・統合処理ラッパー (完全移植)
# =========================================================

def apply_and_save(img_original, filename, face_idx, suffix, roi_list, effect_func, effect_kwargs, face_oval_pts, w_img, h_img):
    base_name, ext = os.path.splitext(filename)
    output_img = img_original.copy()
    transformation_maps = []
    for (x1, y1, x2, y2) in roi_list:
        roi_src = output_img[y1:y2, x1:x2]
        if roi_src.size == 0: continue
        res = effect_func(roi_src, **effect_kwargs)
        processed_roi = res[0]
        map_x, map_y = (res[1], res[2]) if len(res) == 3 else (None, None)
        output_img[y1:y2, x1:x2] = processed_roi
        if map_x is not None: transformation_maps.append(((x1, y1), map_x, map_y))
    new_oval_pts = update_face_oval_points(face_oval_pts, transformation_maps) if transformation_maps else face_oval_pts
    final_bbox = calculate_tight_bbox(new_oval_pts, w_img, h_img)
    save_name = f"{base_name}_{suffix}_{face_idx}"
    cv2.imencode(ext, output_img)[1].tofile(os.path.join(OUTPUT_IMAGE_DIR, save_name + ext))
    save_yolo_format(os.path.join(OUTPUT_LABEL_DIR, save_name + ".txt"), w_img, h_img, final_bbox, 0)

# =========================================================
# 4. エフェクト・ロジック (マップを返す形式)
# =========================================================

def apply_vertical_melt(roi, face_h, power, start_rel, level_rel):
    h, w, _ = roi.shape; y_indices = np.arange(h, dtype=np.float32); face_h = max(face_h, 1.0)
    normalized_y = y_indices / face_h; slope = 1.0 / (level_rel - start_rel + 1e-6)
    curve_y = np.clip((normalized_y - start_rel) * slope, 0.0, 1.0) ** 2 
    x_indices = np.linspace(-1, 1, w, dtype=np.float32); curve_x = np.clip(1.0 - (np.abs(x_indices) ** 4), 0.0, 1.0)
    melt_map = power * (curve_y[:, None] * curve_x[None, :])
    noise = cv2.GaussianBlur(np.random.rand(1, w).astype(np.float32) * 0.1, (0, 0), sigmaX=5)
    melt_map += noise * np.clip(1.0 - (normalized_y - start_rel) * slope, 0.0, 1.0)[:, None] * curve_x[None, :]
    melt_map = np.clip(melt_map, 0.0, 0.99); steps = 1.0 - melt_map
    map_y = np.cumsum(steps, axis=0).astype(np.float32); map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    return cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), map_x, map_y

def apply_squeeze_pinch(roi, face_h, power):
    h, w, _ = roi.shape; cx, cy = w / 2.0, face_h / 2.0 
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32); dx, dy = gx - cx, gy - cy
    radius = np.sqrt(dx**2 + dy**2); max_r = np.sqrt(cx**2 + cy**2) * 1.5 + 1e-6
    factor = np.maximum(0.0, 1.0 - radius / max_r)
    pf = 1.0 - power * (factor ** 2); pf = np.where(pf <= 0.01, 0.01, pf)
    mx, my = (cx + dx / pf).astype(np.float32), (cy + dy / pf).astype(np.float32)
    return cv2.remap(roi, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), mx, my

def apply_eyelid_movement(roi, top_pts, bottom_pts, power=0.6, sigma_scale=2.0):
    h, w = roi.shape[:2]; mx, my = np.meshgrid(np.arange(w), np.arange(h))
    mx, my = mx.astype(np.float32), my.astype(np.float32); all_pts = np.array(top_pts + bottom_pts)
    if len(all_pts) == 0: return roi, mx, my
    cy, eh = np.mean(all_pts[:, 1]), max(1.0, np.max(all_pts[:, 1]) - np.min(all_pts[:, 1]))
    dy = my - cy; sig = eh * sigma_scale; wgt = np.exp(-(dy**2 / (2 * sig**2)))
    my_new = (cy + dy * (1.0 - (power * wgt))).astype(np.float32)
    return cv2.remap(roi, mx, my_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), mx, my_new

def apply_eye_morph(roi, roi_offset, pts_dict, mode='orig_open'):
    h, w = roi.shape[:2]; off_x, off_y = roi_offset; 
    def to_loc(pl): return [(p[0]-off_x, p[1]-off_y) for p in pl]
    working_roi = roi.copy()
    if 'blank' in mode:
        for kt, kb in [('left_eye_top_curve', 'left_eye_bottom_curve'), ('right_eye_top_curve', 'right_eye_bottom_curve')]:
            m = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(m, [np.array(to_loc(pts_dict[kt] + pts_dict[kb][::-1]), np.int32)], 255)
            pix = roi[m > 220]
            if pix.size:
                col = tuple(map(int, np.mean(pix[np.argsort(pix[:,0]*0.114 + pix[:,1]*0.587 + pix[:,2]*0.299)][int(len(pix)*0.5):int(len(pix)*0.9)], 0)))
                mb = cv2.cvtColor(cv2.GaussianBlur(m, (5, 5), 0), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
                working_roi = (working_roi.astype(np.float32)*(1-mb) + np.array(col, np.float32)*mb).astype(np.uint8)
    if 'open' in mode:
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32); bp, bs = random.uniform(0.3, 1.1), random.uniform(1.5, 3.0)
        pl, pr = bp * random.uniform(0.8, 1.2), bp * random.uniform(0.8, 1.2); sl, sr = bs * random.uniform(0.9, 1.1), bs * random.uniform(0.9, 1.1)
        tmp, mx1, my1 = apply_eyelid_movement(working_roi, to_loc(pts_dict['left_eye_top_curve']), to_loc(pts_dict['left_eye_bottom_curve']), pl, sl)
        fin, mx2, my2 = apply_eyelid_movement(tmp, to_loc(pts_dict['right_eye_top_curve']), to_loc(pts_dict['right_eye_bottom_curve']), pr, sr)
        return fin, gx + (mx1-gx) + (mx2-gx), gy + (my1-gy) + (my2-gy)
    return working_roi, None, None

def apply_omni_directional_stretch(roi, pts, off):
    h, w, _ = roi.shape; rel = np.array([(p[0]-off[0], p[1]-off[1]) for p in pts], np.int32)
    cx, cy = np.mean(rel, 0); rel = ((rel - [cx, cy]) * 1.8 + [cx, cy]).astype(np.int32)
    M = cv2.moments(rel); cx, cy = (int(M['m10']/M['m00']), int(M['m01']/M['m00'])) if M['m00']!=0 else (cx, cy)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = gx - cx, gy - cy; dist = np.sqrt(dx**2 + dy**2) + 1e-6
    mx, my = gx + (dx/dist)*(max(rel[:,0])-min(rel[:,0]))*0.6, gy + (dy/dist)*(max(rel[:,1])-min(rel[:,1]))*0.6
    mask = cv2.GaussianBlur(cv2.fillPoly(np.zeros((h,w), np.float32), [rel], 1.0), (21,21), 0)[:,:,None]
    return (cv2.remap(roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)*mask + roi*(1-mask)).astype(np.uint8), mx, my

def apply_fisheye_stretch(roi, power=0.5):
    h, w, _ = roi.shape; cx, cy = w/2.0, h/2.0; gy, gx = np.mgrid[0:h, 0:w].astype(np.float32); dx, dy = gx-cx, gy-cy
    dist = np.sqrt(dx**2+dy**2); mr = min(w, h)/2.0; r_n = dist/mr
    mx, my = cx+dx*(np.power(r_n, power)*mr/(dist+1e-6)), cy+dy*(np.power(r_n, power)*mr/(dist+1e-6))
    mask = r_n > 1.0; mx[mask], my[mask] = gx[mask], gy[mask]
    return cv2.remap(roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), mx, my

def apply_eyeball_pop(roi, pop_strength=2.0):
    h, w, _ = roi.shape; cx, cy = w/2.0, h/2.0; gy, gx = np.mgrid[0:h, 0:w].astype(np.float32); dx, dy = gx-cx, gy-cy
    dist = np.sqrt(dx**2+dy**2); mr = min(w, h)/2.0; r_n = dist/mr
    mx, my = cx+dx*(np.power(r_n, pop_strength)*mr/(dist+1e-6)), cy+dy*(np.power(r_n, pop_strength)*mr/(dist+1e-6))
    mask = r_n > 1.0; mx[mask], my[mask] = gx[mask], gy[mask]
    return cv2.remap(roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), mx, my

def apply_random_suction(roi, target_point, face_size, power): # 引数名を修正
    h, w, _ = roi.shape; cx, cy = target_point; gy, gx = np.mgrid[0:h, 0:w].astype(np.float32); dx, dy = gx-cx, gy-cy
    rad = np.sqrt(dx**2 + dy**2); factor = np.maximum(0.0, 1.0 - rad/(face_size*2.0+1e-6))
    pf = 1.0 - power*(factor**2); pf = np.where(pf <= 0.01, 0.01, pf)
    mx, my = (cx + dx/pf).astype(np.float32), (cy + dy/pf).astype(np.float32)
    return cv2.remap(roi, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), mx, my

# --- テクスチャ用 ---
def create_geometric_mask(roi_shape, face_pts, left_eye_pts, right_eye_pts, lips_pts, left_brow_pts, right_brow_pts, offset):
    h, w = roi_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    off_x, off_y = offset
    def to_roi(pts): return np.array([(p[0]-off_x, p[1]-off_y) for p in pts], dtype=np.int32)

    cv2.fillPoly(mask, [to_roi(face_pts)], 255)
    dilate_size = int(max(w, h) * 0.02)
    if dilate_size < 3: dilate_size = 3
    mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)

    cv2.fillPoly(mask, [to_roi(left_eye_pts)], 0)
    cv2.fillPoly(mask, [to_roi(right_eye_pts)], 0)
    cv2.fillPoly(mask, [to_roi(lips_pts)], 0)

    # 眉毛保護
    brow_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(brow_mask, [to_roi(left_brow_pts)], 255)
    cv2.fillPoly(brow_mask, [to_roi(right_brow_pts)], 255)
    erode_size = int(max(w, h) * 0.01) 
    if erode_size < 1: erode_size = 1
    brow_mask = cv2.erode(brow_mask, np.ones((erode_size, erode_size), np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(brow_mask))
    return mask

def create_advanced_skin_mask_with_position(roi, face_landmarks, w_img, h_img, offset, eye_level_y_rel):
    h_roi, w_roi = roi.shape[:2]
    off_x, off_y = offset
    cfg = COLOR_THRESHOLDS

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    
    sampled_hsvs = []
    sampled_ycrcbs = []
    for idx in SKIN_SAMPLE_INDICES:
        lm = face_landmarks.landmark[idx]
        lx, ly = int(lm.x * w_img) - off_x, int(lm.y * h_img) - off_y
        if 0 <= lx < w_roi and 0 <= ly < h_roi:
            y_min, y_max = max(0, ly-2), min(h_roi, ly+3)
            x_min, x_max = max(0, lx-2), min(w_roi, lx+3)
            if y_max > y_min and x_max > x_min:
                sampled_hsvs.append(np.mean(hsv_roi[y_min:y_max, x_min:x_max], axis=(0, 1)))
                sampled_ycrcbs.append(np.mean(ycrcb_roi[y_min:y_max, x_min:x_max], axis=(0, 1)))
    
    if not sampled_hsvs: return np.ones((h_roi, w_roi), dtype=np.uint8) * 255

    mean_hsv = np.mean(sampled_hsvs, axis=0)
    mean_ycrcb = np.mean(sampled_ycrcbs, axis=0)
    
    lower_hsv = np.array([max(0, mean_hsv[0]-cfg['h_tol']), max(0, mean_hsv[1]-cfg['s_tol_lower']), max(0, mean_hsv[2]-cfg['v_tol_lower'])]).astype(np.uint8)
    upper_hsv = np.array([min(180, mean_hsv[0]+cfg['h_tol']), min(255, mean_hsv[1]+cfg['s_tol_upper']), min(255, mean_hsv[2]+cfg['v_tol_upper'])]).astype(np.uint8)   
    # その下の YCrCb も同様に修正しておくと安心です
    lower_ycrcb = np.array([0, max(0, mean_ycrcb[1]-cfg['cr_tol']), max(0, mean_ycrcb[2]-cfg['cb_tol'])]).astype(np.uint8)
    upper_ycrcb = np.array([255, min(255, mean_ycrcb[1]+cfg['cr_tol']), min(255, mean_ycrcb[2]+cfg['cb_tol'])]).astype(np.uint8)
    mask_hsv_base = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
    mask_ycrcb_base = cv2.inRange(ycrcb_roi, lower_ycrcb, upper_ycrcb)
    
    base_mask = cv2.bitwise_and(mask_hsv_base, mask_ycrcb_base)
    
    v_channel = hsv_roi[:, :, 2]
    _, mask_highlight = cv2.threshold(v_channel, cfg['highlight_threshold'], 255, cv2.THRESH_BINARY)

    grid_y, _ = np.meshgrid(np.arange(h_roi), np.arange(w_roi), indexing='ij')
    mask_area_upper = (grid_y < eye_level_y_rel).astype(np.uint8) * 255
    mask_area_lower = (grid_y >= eye_level_y_rel).astype(np.uint8) * 255
    
    s_channel = hsv_roi[:, :, 1]
    mask_sat_strict = cv2.inRange(s_channel, cfg['min_saturation_strict'], 255)
    
    upper_standard = cv2.bitwise_and(base_mask, mask_sat_strict)
    upper_highlight_recovery = cv2.bitwise_and(mask_highlight, mask_ycrcb_base)
    upper_final = cv2.bitwise_or(upper_standard, upper_highlight_recovery)
    upper_result = cv2.bitwise_and(upper_final, mask_area_upper)
    
    lower_final = cv2.bitwise_or(base_mask, mask_highlight)
    lower_result = cv2.bitwise_and(lower_final, mask_area_lower)
    
    final_mask = cv2.bitwise_or(upper_result, lower_result)
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return final_mask

def create_hybrid_mask(geo_mask, skin_mask, eye_level_y):
    h, w = geo_mask.shape
    grid_y, _ = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    mask_upper_area = (grid_y < eye_level_y).astype(np.uint8) * 255
    mask_lower_area = (grid_y >= eye_level_y).astype(np.uint8) * 255
    upper_result = cv2.bitwise_and(geo_mask, skin_mask)
    upper_result = cv2.bitwise_and(upper_result, mask_upper_area)
    lower_result = cv2.bitwise_and(geo_mask, mask_lower_area)
    final_mask = cv2.bitwise_or(upper_result, lower_result)
    final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
    return final_mask

def apply_skin_color_change(roi, mask, color_mode):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv_roi)

    if color_mode == 'red':
        h[:] = 175
        s_factor = random.uniform(3, 5)
        s = s * s_factor
        s = np.clip(s, 0, 255)
    elif color_mode == 'green':
        h[:] = 55
        s_factor = random.uniform(3, 5)
        s = np.maximum(s * s_factor, random.uniform(30, 60))
        s = np.clip(s, 0, 255)
    elif color_mode == 'blue':
        h[:] = 110 
        s_factor = random.uniform(3, 5)
        s = np.maximum(s * s_factor, random.uniform(40, 80))
        s = np.clip(s, 0, 255)
    elif color_mode == 'dark':
        v_factor = random.uniform(0.05, 0.4)
        s_factor = random.uniform(0.05, 0.3)
        s = s * s_factor
        v = v * v_factor
    elif color_mode == 'bleach':
        # 彩度(s)をほぼゼロにして色味を消す
        s_factor = random.uniform(0.0, 0.1)
        s = s * s_factor
        # 明度(v)を大幅に上げて白く飛ばす
        # 元の明るさにプラスすることで、ハイライト感を強調
        v_boost = random.uniform(50, 100) 
        v = np.clip(v + v_boost, 0, 255)

    merged_hsv = cv2.merge([h, s, v])
    colored_roi = cv2.cvtColor(merged_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    roi_float = roi.astype(np.float32)
    colored_roi_float = colored_roi.astype(np.float32)
    
    # マスク適用
    mask_3ch = cv2.merge([mask, mask, mask])
    output = roi_float * (1.0 - mask_3ch) + colored_roi_float * mask_3ch
    return output.astype(np.uint8)

# --- Gen 8: Skin Noise ---
def apply_skin_noise(roi, mask, noise_mode):
    h, w, c = roi.shape
    roi_float = roi.astype(np.float32)
    noise_img = np.zeros_like(roi_float)
    if noise_mode == 'grain':
        sigma = random.uniform(20, 80)
        scale = random.uniform(0.15, 0.4) 
        small_h, small_w = int(h * scale), int(w * scale)
        if small_h < 1: small_h = 1
        if small_w < 1: small_w = 1
        gauss_small = np.random.normal(0, sigma, (small_h, small_w)).astype(np.float32)
        gauss = cv2.resize(gauss_small, (w, h), interpolation=cv2.INTER_LINEAR)
        noise_img[:, :, 0] = gauss
        noise_img[:, :, 1] = gauss
        noise_img[:, :, 2] = gauss
    elif noise_mode == 'digital':
        sigma = random.uniform(30, 100)
        scale = random.uniform(0.05, 0.25)
        small_h, small_w = int(h * scale), int(w * scale)
        if small_h < 1: small_h = 1
        if small_w < 1: small_w = 1
        noise_small = np.random.normal(0, sigma, (small_h, small_w, c)).astype(np.float32)
        noise_img = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_NEAREST)
    noisy_roi = cv2.add(roi_float, noise_img)
    noisy_roi = np.clip(noisy_roi, 0, 255).astype(np.uint8)
    mask_3ch = cv2.merge([mask, mask, mask])
    output = noisy_roi.astype(np.float32) * mask_3ch + roi_float * (1.0 - mask_3ch)
    return output.astype(np.uint8)

# --- Gen 9: Texture ---
def apply_texture_pattern(roi, mask, texture_img):
    h, w = roi.shape[:2]
    th, tw = texture_img.shape[:2]
    if th < h or tw < w:
        scale = max(h / th, w / tw) * 1.5 
        texture_img = cv2.resize(texture_img, None, fx=scale, fy=scale)
        th, tw = texture_img.shape[:2]
    x_start = random.randint(0, tw - w)
    y_start = random.randint(0, th - h)
    crop = texture_img[y_start:y_start+h, x_start:x_start+w]
    gray_tex = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex_mean = np.mean(gray_tex)
    tex_diff = gray_tex - tex_mean
    strength = random.uniform(0.5, 1.5)
    tex_diff = tex_diff * strength
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv_roi)
    v_new = v_ch + tex_diff
    v_new = np.clip(v_new, 0, 255)
    hsv_new = cv2.merge([h_ch, s_ch, v_new])
    textured_roi = cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
    mask_3ch = cv2.merge([mask, mask, mask])
    roi_float = roi.astype(np.float32)
    tex_float = textured_roi.astype(np.float32)
    output = tex_float * mask_3ch + roi_float * (1.0 - mask_3ch)
    return output.astype(np.uint8)
# =========================================================
# 5. メイン処理
# =========================================================

def process_images():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    textures = [(os.path.splitext(f)[0], cv2.imread(os.path.join(TEXTURE_DIR, f))) for f in os.listdir(TEXTURE_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]
    mp_face_mesh, mp_selfie, mp_face_det = mp.solutions.face_mesh, mp.solutions.selfie_segmentation, mp.solutions.face_detection
    
    image_counter = 0
    face_counter = 0
    not_face_counter = 0
    task_stats = {}  # ここに { 'shadow': 10, 'bleach': 5, ... } と記録される

    with mp_selfie.SelfieSegmentation(model_selection=1) as selfie_seg, \
         mp_face_mesh.FaceMesh(static_image_mode=True,  max_num_faces=10, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh, \
         mp_face_det.FaceDetection(model_selection=1) as face_det:
        
        for filename in files:
            image_counter += 1
            base_n, ext = os.path.splitext(filename)
            if os.path.exists(os.path.join(OUTPUT_IMAGE_DIR, f"{base_n}_1_0{ext}")):
                print(f" \n [Skip] 既に処理済み: {filename} ([{image_counter}/{len(files)}])")
                continue # すでに存在すれば飛ばす


            print(f"\n{'='*30}")
            print(f" 処理開始: {filename} ([{image_counter}/{len(files)}])")
            print(f"{'='*30}")
            img_original = cv2.imdecode(np.fromfile(os.path.join(INPUT_DIR, filename), np.uint8), cv2.IMREAD_COLOR)
            if img_original is None:
                print(f" [Error] 画像を読み込めませんでした: {filename}")
                continue
            h_i, w_i = img_original.shape[:2]; img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            res_seg, res_mesh, res_det = selfie_seg.process(img_rgb), face_mesh.process(img_rgb), face_det.process(img_rgb)
            # 解析の直前
            img_for_mesh = np.ascontiguousarray(img_rgb.copy())
            res_mesh = face_mesh.process(img_for_mesh)
            mp_face_mesh = mp.solutions.face_mesh
            mp_selfie = mp.solutions.selfie_segmentation
            mp_face_det = mp.solutions.face_detection
            if not (res_mesh.multi_face_landmarks and res_det.detections):
                print(f" [Skip] 顔が検出されなかったため、処理をスキップしました") 
                not_face_counter += 1
                continue
            
            face_counter +=  1
            person_mask = (res_seg.segmentation_mask > 0.5).astype(np.uint8) * 255

            for i, (lms, det) in enumerate(zip(res_mesh.multi_face_landmarks, res_det.detections)):
                pts = {k: [(int(lms.landmark[idx].x*w_i), int(lms.landmark[idx].y*h_i)) for idx in v] if isinstance(v, list) else (int(lms.landmark[v].x*w_i), int(lms.landmark[v].y*h_i)) for k, v in LM_INDICES.items()}
                
                xs_all = [lm.x for lm in lms.landmark]; ys_all = [lm.y for lm in lms.landmark]
                f_xmin, f_xmax = int(min(xs_all) * w_i), int(max(xs_all) * w_i)
                raw_ymin, raw_ymax = int(min(ys_all) * h_i), int(max(ys_all) * h_i)
                fw, fh = f_xmax - f_xmin, raw_ymax - raw_ymin
                f_cx, f_cy = (f_xmin + f_xmax) // 2, (raw_ymin + raw_ymax) // 2
                face_bbox = calculate_tight_bbox(pts['face_oval'], w_i, h_i)
                base_n, ext = os.path.splitext(filename)

                lama_tasks = ["headless", "eyes", "two_wide_rects"]
                geo_tasks = ["melt", "squeeze", "blank_only", "blank_open", "orig_open", "eyes_hidden", "mouth_hidden", "circle_fisheye", "circle_pop_out", "warp_squeeze"]
                tex_tasks = ["bleach","dark", "red", "green", "blue", "grain", "digital"] + [f"tex_{t[0]}" for t in textures]
                selected_tasks = lama_tasks + geo_tasks + tex_tasks

                # --- 1. LaMa系 (class 2, 1) 実行セクション ---
                active_lama_tasks = [x for x in selected_tasks if x in lama_tasks]

                if active_lama_tasks:
                    # 推論のためのマスク準備 (頭部領域の特定)
                    bbox_rel = det.location_data.relative_bounding_box
                    y_cut = int(bbox_rel.ymin * h_i) + int(bbox_rel.height * h_i) + int(int(bbox_rel.height * h_i) * 0.1)
                    x_start = max(0, int(bbox_rel.xmin * w_i) - int(int(bbox_rel.width * w_i) * 0.8))
                    x_end = min(w_i - 1, int(bbox_rel.xmin * w_i) + int(bbox_rel.width * w_i) + int(int(bbox_rel.width * w_i) * 0.8))
                    
                    h_m = np.zeros((h_i, w_i), np.uint8)
                    h_m[:y_cut, x_start:x_end] = person_mask[:y_cut, x_start:x_end]
                    h_m = cv2.dilate(h_m, np.ones((7, 7), np.uint8), iterations=2)
                    cv2.circle(h_m, (int((bbox_rel.xmin + bbox_rel.width / 2) * w_i), y_cut), int(int(bbox_rel.height * h_i) * 0.2), 255, -1)
                    h_m[y_cut:, :] = 0
                    
                    # LaMa推論実行 (一番重い処理)
                    res_lama = model(img_rgb, h_m, InpaintRequest(hd_strategy=HDStrategy.ORIGINAL))
                    bg_img = np.clip(res_lama if res_lama.max() > 1 else res_lama * 255, 0, 255).astype(np.uint8)
                    
                    # 頭部が消えたベース画像 (headless状態)
                    hless_base = bg_img.copy()
                    hless_base[cv2.bitwise_not(h_m) == 255] = img_original[cv2.bitwise_not(h_m) == 255]

                    for t in active_lama_tasks:
                        task_stats[t] = task_stats.get(t, 0) + 1
                        out_lama = hless_base.copy()
                        cid = 2 if t == "headless" else 1
                        
                        if t == "eyes":
                            # 目だけを元の画像から戻す
                            em = np.zeros((h_i, w_i), np.uint8)
                            for ir, ck in [(468, 'left_eye_contour'), (473, 'right_eye_contour')]:
                                p = (int(lms.landmark[ir].x * w_i), int(lms.landmark[ir].y * h_i))
                                # 目の半径を計算して円を描画
                                r = int(math.sqrt((pts[ck][8][0] - pts[ck][0][0])**2 + (pts[ck][8][1] - pts[ck][0][1])**2) / 2)
                                cv2.circle(em, p, r, 255, -1)
                            out_lama[em == 255] = img_original[em == 255]
                            
                        elif t == "two_wide_rects":
                            rm = h_m.copy()
                            yc, xc = np.where(h_m == 255)
                            if len(xc) > 0:
                                # 顔の中心座標を計算
                                face_center_x, face_center_y = f_cx, f_cy
                                for _ in range(2):
                                    idx_r = random.randint(0, len(xc) - 1)
                                    # ★ 矩形の中心を「ランダムな点」から「少し顔の中心寄り」に補正(30%引き寄せる)
                                    target_x = int(xc[idx_r] * 0.7 + face_center_x * 0.3)
                                    target_y = int(yc[idx_r] * 0.7 + face_center_y * 0.3)
                                    
                                    rw = random.randint(int(fw * 0.5), int(fw * 0.7))
                                    rh = random.randint(int(fh * 0.3), int(fh * 0.5))
                                    cv2.rectangle(rm, (target_x - rw // 2, target_y - rh // 2), 
                                                 (target_x + rw // 2, target_y + rh // 2), 0, -1)
                        
                                out_lama[cv2.bitwise_and(rm, h_m) == 255] = img_original[cv2.bitwise_and(rm, h_m) == 255]
                        
                        # 保存処理
                        save_name = f"{base_n}_{t}_{i}"
                        cv2.imencode(ext, out_lama)[1].tofile(os.path.join(OUTPUT_IMAGE_DIR, save_name + ext))
                        save_yolo_format(os.path.join(OUTPUT_LABEL_DIR, save_name + ".txt"), w_i, h_i, face_bbox, cid)

                # --- 2. Geometry系 実行セクション ---
                # 共通変数の準備
                f_cx, f_cy = (f_xmin + f_xmax) // 2, (raw_ymin + raw_ymax) // 2

                for t in [x for x in selected_tasks if x in geo_tasks]:
                    task_stats[t] = task_stats.get(t, 0) + 1
                    # --- Gen 2: Melt / Squeeze ---
                    if t in ['melt', 'squeeze']:
                        r_gen2 = [get_roi_rect(f_cx, f_cy, fw, fh, 2.5, w_i, h_i)]
                        f_chin_y_rel = pts['bottom'][1] - r_gen2[0][1]
                        rnd_pow = random.uniform(0.4, 0.7) if t == 'melt' else random.uniform(0.4, 0.95)
                        func = apply_vertical_melt if t == 'melt' else apply_squeeze_pinch
                        kw = {'face_h': f_chin_y_rel, 'power': rnd_pow}
                        if t == 'melt':
                            v1, v2 = random.uniform(0.4, 0.95), random.uniform(0.4, 0.95)
                            kw.update({'start_rel': min(v1, v2), 'level_rel': max(v1, v2)})
                        apply_and_save(img_original, filename, i, t, r_gen2, func, kw, pts['face_oval'], w_i, h_i)

                    # --- Gen 3: Eye Morph ---
                    elif t in ['blank_only', 'blank_open', 'orig_open']:
                        r_g3 = [get_roi_rect(f_cx, f_cy, fw, fh, 0.4, w_i, h_i)]
                        apply_and_save(img_original, filename, i, t, r_g3, apply_eye_morph, 
                                       {'pts_dict': pts, 'roi_offset': (r_g3[0][0], r_g3[0][1]), 'mode': t}, 
                                       pts['face_oval'], w_i, h_i)

                    # --- Gen 4: Hidden Stretch ---
                    elif t in ["eyes_hidden", "mouth_hidden"]:
                        keys = ['left_eye_contour', 'right_eye_contour'] if t == "eyes_hidden" else ['lips_contour']
                        t_pts = []
                        for k in keys: t_pts.extend(pts[k])
                        txs, tys = [p[0] for p in t_pts], [p[1] for p in t_pts]
                        hx1, hy1, hx2, hy2 = get_roi_rect(int(np.mean(txs)), int(np.mean(tys)), 
                                                         max(txs)-min(txs), max(tys)-min(tys), 1.0, w_i, h_i)
                        
                        def apply_multi_omni_wrapper(roi_in, key_list, pts_dict, offset):
                            r = roi_in.copy()
                            for k in key_list:
                                r, _, _ = apply_omni_directional_stretch(r, pts_dict[k], offset)
                            return r, None, None
                            
                        apply_and_save(img_original, filename, i, t, [(hx1, hy1, hx2, hy2)], 
                                       apply_multi_omni_wrapper, {'key_list': keys, 'pts_dict': pts, 'offset': (hx1, hy1)}, 
                                       pts['face_oval'], w_i, h_i)

                    # --- Gen 5: Fisheye / Pop Out ---
                    elif t in ['circle_fisheye', 'circle_pop_out']:
                        mode = 'fisheye' if 'fisheye' in t else 'pop_out'
                        rois = []
                        for k in ['left_eye_contour', 'right_eye_contour']:
                            ex, ey = [p[0] for p in pts[k]], [p[1] for p in pts[k]]
                            rois.append(get_roi_rect(int(np.mean(ex)), int(np.mean(ey)), 
                                                     max(ex)-min(ex), max(ey)-min(ey), 
                                                     2.5 if mode=='fisheye' else 3.2, w_i, h_i))
                        func = apply_fisheye_stretch if mode=='fisheye' else apply_eyeball_pop
                        kw = {'power': random.uniform(0.3, 0.6)} if mode=='fisheye' else {'pop_strength': random.uniform(1.8, 3.0)}
                        apply_and_save(img_original, filename, i, t, rois, func, kw, pts['face_oval'], w_i, h_i)

                    # --- Gen 6: Warp Squeeze ---
                    elif t == "warp_squeeze":
                        sl = int(max(fw, fh) * 1.8)
                        r_g6 = get_roi_rect(f_cx, f_cy, sl, sl, 0, w_i, h_i)
                        ang = random.uniform(0, 2 * math.pi)
                        dist = max(fw, fh) * random.uniform(0, 0.8)
                        t_pt = (f_cx + dist * math.cos(ang) - r_g6[0], f_cy + dist * math.sin(ang) - r_g6[1])
                        apply_and_save(img_original, filename, i, t, [r_g6], apply_random_suction, 
                                       {'target_point': t_pt, 'face_size': max(fw, fh), 'power': random.uniform(0.6, 0.95)}, 
                                       pts['face_oval'], w_i, h_i)
                # --- 3. テクスチャ (class 3) ---
                pts = {}
                for key, val in LM_INDICES.items():
                    if isinstance(val, list):
                        pts[key] = [(int(lms.landmark[idx].x * w_i), int(lms.landmark[idx].y * h_i)) for idx in val]
                    else:
                        pts[key] = [(int(lms.landmark[val].x * w_i), int(lms.landmark[val].y * h_i))]

              
                
                # --- 3. Texture系 実行セクション ---
                # Texture系のタスクが一つでも選ばれているか確認
                active_tex_tasks = [x for x in selected_tasks if x in tex_tasks]

                if active_tex_tasks:
                    # --- [共通準備] マスク (fmn) の生成 ---
                    # おでこ拡張を含む顔の高さ計算
                    margin_up_tex = int(fh * FOREHEAD_EXTEND_RATIO)
                    face_h_tex = raw_ymax - (raw_ymin - margin_up_tex)

                    # ROI切り出し (Geometry系とはマージン設定が異なるため再計算)
                    px_t, py_t = int(fw * 0.5), int(face_h_tex * 0.5)
                    sx1, sy1 = max(0, f_xmin - px_t), max(0, (raw_ymin - margin_up_tex) - py_t)
                    sx2, sy2 = min(w_i, f_xmax + px_t), min(h_i, raw_ymax + py_t)
                    
                    roi_skin = img_original[sy1:sy2, sx1:sx2]

                    if roi_skin.size > 0:
                        # 目線の高さ
                        eye_ys = [p[1] for p in pts['left_eye_contour'] + pts['right_eye_contour']]
                        eye_lvl = max(0, min(eye_ys) - sy1) if eye_ys else roi_skin.shape[0] // 3
                        
                        # おでこ拡張座標
                        f_pts_ext = extend_forehead_points(pts['face_oval'], face_h_tex, 0.15)

                        # 各種マスク作成と合成
                        g_m = create_geometric_mask(roi_skin.shape, f_pts_ext, 
                                                    pts['left_eye_contour'], pts['right_eye_contour'], 
                                                    pts['lips_inner'], pts['left_eyebrow'], 
                                                    pts['right_eyebrow'], (sx1, sy1))
                        s_m = create_advanced_skin_mask_with_position(roi_skin, lms, w_i, h_i, (sx1, sy1), eye_lvl)
                        fmn = create_hybrid_mask(g_m, s_m, eye_lvl).astype(np.float32) / 255.0

                        # --- 個別のエフェクト適用 ---
                        for t in active_tex_tasks:
                            task_stats[t] = task_stats.get(t, 0) + 1
                            out_tex = img_original.copy()
                            
                            if t in ['red', 'green', 'blue', 'dark', 'bleach']:
                                # 肌色変更
                                processed_roi = apply_skin_color_change(roi_skin, fmn, t)
                                save_suffix = t
                            elif t in ['grain', 'digital']:
                                # ノイズ
                                processed_roi = apply_skin_noise(roi_skin, fmn, t)
                                save_suffix = t
                            elif t.startswith("tex_"):
                                # テクスチャ画像適用
                                tex_id = t.replace("tex_", "")
                                # texturesリストから該当する画像を探す
                                t_img = next((img for name, img in textures if name == tex_id), None)
                                if t_img is not None:
                                    processed_roi = apply_texture_pattern(roi_skin, fmn, t_img)
                                    save_suffix = tex_id
                                else:
                                    continue
                            
                            # 画像保存とYOLO書き出し
                            out_tex[sy1:sy2, sx1:sx2] = processed_roi
                            p_n = f"{base_n}_{save_suffix}_{i}"
                            cv2.imencode(ext, out_tex)[1].tofile(os.path.join(OUTPUT_IMAGE_DIR, p_n + ext))
                            save_yolo_format(os.path.join(OUTPUT_LABEL_DIR, p_n + ".txt"), w_i, h_i, face_bbox, 3)
            if device.type == 'cuda':
                import torch
                torch.cuda.empty_cache()  # キャッシュを空にする
                import gc; gc.collect() # Python側のゴミ掃除（必要なら）
    print("\n\n全工程完了。")
    
    print(f"\n{'='*30}")
    print(f"リザルト")
    print(f"{'='*30}")
    print(f"画像総数:{image_counter}\n\n処理成功:{face_counter}\nスキップ:{not_face_counter}")
    print(f"\n処理率:{face_counter/image_counter*100}%")
    # print(f"\n[ エフェクト別実行数 ]")
    # # 実行数が多い順に並べ替えて表示
    # sorted_tasks = sorted(task_stats.items(), key=lambda x: x[1], reverse=True)
    # for task_name, count in sorted_tasks:
    #     print(f" - {task_name:15}: {count} 回")
    print(f"{'='*30 }")
    print()

if __name__ == "__main__": process_images()