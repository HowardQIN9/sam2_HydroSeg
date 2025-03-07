import os
import re
import cv2
import json
import numpy as np
import torch
from datetime import datetime, timedelta
from segment_anything import SamPredictor, sam_model_registry

def get_previous_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    prev_date_obj = date_obj - timedelta(days=1)
    return prev_date_obj.strftime("%Y%m%d")

def parse_small_image_filename(filename):
    pattern = r"image_(\d{8})_\d{6}_(T\d+[A-Z]\d+)_(\d+)_(\d+)_(\d+)_(\d+)_padded\.jpg"
    match = re.match(pattern, filename)
    if match:
        date_str, obj_id, x_min, y_min, x_max, y_max = match.groups()
        crop_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        return date_str, obj_id, crop_bbox
    return None, None, None

def get_previous_bbox(bbox_json_path, prev_date, obj_id):
    """
    获取前一天相同 obj_id 的 bounding box
    """
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    
    for filename, file_data in bbox_data.items():
        if prev_date in filename:
            # Debug: 查看 file_data 真实结构
            print(f"[DEBUG] Checking {filename}: {file_data}")

            if isinstance(file_data, list):  # 处理 file_data 为列表的情况
                for bbox_entry in file_data:
                    if bbox_entry.get("label") == obj_id:  # 用 get() 避免 KeyError
                        return bbox_entry.get("bbox")  # 返回 bbox 数组
            elif isinstance(file_data, dict) and "bboxes" in file_data:  # 兼容可能的字典格式
                for bbox_entry in file_data["bboxes"]:
                    if bbox_entry.get("label") == obj_id:
                        return bbox_entry.get("bbox")
    
    print(f"[WARNING] No BBox found for object {obj_id} on {prev_date}")
    return None  # 没找到匹配的 bbox


def map_bbox_to_cropped_image(prev_bbox, crop_bbox):
    px_min, py_min, px_max, py_max = prev_bbox
    cx_min, cy_min, cx_max, cy_max = crop_bbox
    
    new_x_min = max(0, px_min - cx_min)
    new_y_min = max(0, py_min - cy_min)
    new_x_max = min(cx_max - cx_min, px_max - cx_min)
    new_y_max = min(cy_max - cy_min, py_max - cy_min)
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)

def run_sam_on_cropped_image(small_image_path, bbox_prompt, sam_model):
    img = cv2.imread(small_image_path)
    if img is None:
        print(f"[ERROR] Failed to read: {small_image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    predictor = SamPredictor(sam_model)
    predictor.set_image(img_rgb)

    bbox_prompt = np.array(bbox_prompt, dtype=np.float32).reshape(1, 4)

    masks, _, _ = predictor.predict(box=bbox_prompt)
    return masks[0]

def run_sam_on_rois(small_image_folder, bbox_json_path, sam_checkpoint, model_type="vit_h"):
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

    for filename in os.listdir(small_image_folder):
        if not filename.endswith("_padded.jpg"):
            continue
        
        date_str, obj_id, crop_bbox = parse_small_image_filename(filename)
        if not date_str or not obj_id:
            print(f"[WARNING] Failed to parse filename: {filename}")
            continue

        prev_date = get_previous_date(date_str)
        print(f"\n[INFO] Processing: {filename}, Object ID: {obj_id}, Date: {date_str} -> {prev_date}")

        prev_bbox = get_previous_bbox(bbox_json_path, prev_date, obj_id)
        if prev_bbox is None:
            print(f"[WARNING] No BBox found for previous day: {obj_id}")
            continue

        new_bbox = map_bbox_to_cropped_image(prev_bbox, crop_bbox)
        print(f"[INFO] Previous BBox: {prev_bbox} -> Cropped Image BBox: {new_bbox}")

        small_image_path = os.path.join(small_image_folder, filename)
        mask = run_sam_on_cropped_image(small_image_path, new_bbox, sam)

        if mask is not None:
            # 确保 mask 转换为 uint8
            mask = (mask * 255).astype(np.uint8)

            # Apply post-processing: erosion, dilation, and largest region selection
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                filtered_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            else:
                filtered_mask = mask

            mask_save_path = small_image_path.replace("_padded.jpg", "_mask.png")
            cv2.imwrite(mask_save_path, filtered_mask)
            print(f"[SUCCESS] Saved segmentation mask: {mask_save_path}")




