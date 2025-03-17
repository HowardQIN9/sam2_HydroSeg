import os
import re
import cv2
import json
import numpy as np
import torch
from datetime import datetime, timedelta
from segment_anything import SamPredictor, sam_model_registry

def compute_iou(bbox1, bbox2):
    """计算 Intersection over Union (IoU)"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0  # 避免 ZeroDivisionError

def get_previous_date(date_str):
    """返回前一天的 YYYYMMDD 格式"""
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    prev_date_obj = date_obj - timedelta(days=1)
    return prev_date_obj.strftime("%Y%m%d")

def extract_date(filename):
    """从文件名提取日期"""
    match = re.match(r"^image_(\d{8})_\d{6}\.jpg$", filename)
    return match.group(1) if match else None

def parse_small_image_filename(filename):

    pattern = r"image_(\d{8})_\d{6}_(T\d+[A-Z]\d+)_(\d+)_(\d+)_(\d+)_(\d+)_cropped\.jpg"
    match = re.match(pattern, filename)
    if match:
        date_str, obj_id, x_min, y_min, x_max, y_max = match.groups()
        crop_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        return date_str, obj_id, crop_bbox
    return None, None, None

def get_previous_bbox(bbox_json_path, prev_date, obj_id):
    """获取前一天相同 obj_id 的 bounding box"""
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    
    for filename, file_data in bbox_data.items():
        if prev_date in filename:
            if isinstance(file_data, list):
                for bbox_entry in file_data:
                    if bbox_entry.get("label") == obj_id:
                        return bbox_entry.get("bbox")
            elif isinstance(file_data, dict) and "bboxes" in file_data:
                for bbox_entry in file_data["bboxes"]:
                    if bbox_entry.get("label") == obj_id:
                        return bbox_entry.get("bbox")
    
    return None  # 没找到匹配的 bbox

def map_bbox_to_cropped_image(prev_bbox, crop_bbox):
    """将 `prev_bbox` 映射到 `crop_bbox` 内"""
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
        print(f"[ERROR] Image not found: {small_image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    predictor = SamPredictor(sam_model)
    predictor.set_image(img_rgb)

    bbox_prompt = np.array(bbox_prompt, dtype=np.float32).reshape(1, 4)

    masks, _, _ = predictor.predict(box=bbox_prompt)

    if masks is not None and masks.shape[0] > 0:
        return masks[0]
    else:
        print(f"[WARNING] No mask generated for {small_image_path}")
        return None


def process_bad_cases(bad_case_json, bbox_json_path, small_image_folder, save_folder, crop_output_dir, sam_checkpoint, model_type):
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(crop_output_dir, exist_ok=True)
    
    with open(bad_case_json, "r", encoding="utf-8") as f:
        bad_cases = json.load(f)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

    corrected_bboxes = {}
    for filename in os.listdir(small_image_folder):
        
        if not filename.endswith("_cropped.jpg"):
            continue
        
        date_str, obj_id, crop_bbox = parse_small_image_filename(filename)
        if not date_str or not obj_id:
            continue

        prev_date = get_previous_date(date_str)
        prev_bbox = get_previous_bbox(bbox_json_path, prev_date, obj_id)
        if prev_bbox is None:
            continue
        print(f"Processing: {filename}, obj_id: {obj_id}, crop_bbox: {crop_bbox}")
        new_bbox = map_bbox_to_cropped_image(prev_bbox, crop_bbox)
        small_image_path = os.path.join(small_image_folder, filename)
        mask = run_sam_on_cropped_image(small_image_path, new_bbox, sam)

        if mask is not None:
            mask = (mask * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

            mask_filename = filename.replace("_cropped.jpg", "_mask.jpg")
            mask_save_path = os.path.join(crop_output_dir, mask_filename)
            cv2.imwrite(mask_save_path, mask)
            print(f"[INFO] Mask saved: {mask_save_path}")

    
    return corrected_bboxes






def main():
    points_json = "/home/zqin74/RGB/point_prompts9.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    image_root_dir = "/home/zqin74/RGB/Rasp3"
    output_dir = "/home/zqin74/RGB/v2/Seg_Rap3"
    mask_json_output = "/home/zqin74/RGB/v2/masks3.json"  # **存储最终所有 mask**
    bad_case_json_output = "/home/zqin74/RGB/v2/bad_case3.json"
    crop_output_dir = "/home/zqin74/RGB/v2/crop3"
    projected_mask_output_dir = "/home/zqin74/RGB/v2/projected_masks"
    sam_model_type = "vit_h"
    print("[INFO] Starting bad case correction loop...")
    process_bad_cases(
    bad_case_json=bad_case_json_output,
    bbox_json_path=mask_json_output,  # 这里应该是 bbox_json_path，而不是 masks_json_path
    small_image_folder=crop_output_dir,
    save_folder=crop_output_dir,
    crop_output_dir=crop_output_dir,
    sam_checkpoint=sam_checkpoint_path,
    model_type=sam_model_type
    )



if __name__ == "__main__":
    main()
