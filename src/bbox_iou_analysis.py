import os
import re
import cv2
import json
from datetime import datetime, timedelta

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
    return inter_area / union_area if union_area > 0 else 0

def get_previous_date(date_str):
    """返回前一天的 YYYYMMDD 格式"""
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    prev_date_obj = date_obj - timedelta(days=1)
    return prev_date_obj.strftime("%Y%m%d")

def extract_date(filename):
    """从文件名提取日期"""
    match = re.match(r"^image_(\d{8})_\d{6}\.jpg$", filename)
    return match.group(1) if match else None

def gather_bboxes_for_day(bbox_data, date_yyyymmdd):
    """收集某一天的所有 bbox 信息，返回 {label: bbox} 形式的字典"""
    label2bbox = {}
    pattern_day = f"^image_{date_yyyymmdd}_\\d{{6}}\\.jpg$"

    for filename, file_data in bbox_data.items():
        if re.match(pattern_day, filename):
            if isinstance(file_data, list):  # 兼容列表格式
                bbox_entries = file_data
            elif isinstance(file_data, dict) and "bboxes" in file_data:
                bbox_entries = file_data["bboxes"]
            else:
                print(f"[ERROR] Unexpected format for {filename}: {type(file_data)}")
                continue  # 跳过错误数据
            
            for bbox_entry in bbox_entries:
                label = bbox_entry.get("label", None)
                if label and (label not in label2bbox):
                    label2bbox[label] = bbox_entry["bbox"]
    
    return label2bbox


def process_missing_bboxes(bad_case_json, bbox_json_path, image_folder, save_folder, crop_folder, json_output_path, iou_threshold=0.8, padding=10):
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(crop_folder, exist_ok=True)
    
    with open(bad_case_json, "r", encoding="utf-8") as f:
        bad_cases = json.load(f)
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    
    missing_bboxes = {}
    for img_basename, current_bboxes in bad_cases.items():
        date_str = extract_date(img_basename)
        if not date_str:
            continue
        prev_date = get_previous_date(date_str)
        prev_label2bbox = gather_bboxes_for_day(bbox_data, prev_date)
        for cur_bbox_entry in current_bboxes:
            c_label = cur_bbox_entry.get("label", None)
            c_bbox = cur_bbox_entry["bbox"]
            if not c_label or c_label not in prev_label2bbox:
                missing_bboxes.setdefault(img_basename, []).append({"bbox": c_bbox, "label": c_label or "Unknown"})
                continue
            prev_bbox = prev_label2bbox[c_label]
            iou = compute_iou(c_bbox, prev_bbox)
            if iou < iou_threshold:
                missing_bboxes.setdefault(img_basename, []).append({"bbox": c_bbox, "label": c_label})
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(missing_bboxes, f, indent=4)
    
    for img_basename, bboxes in missing_bboxes.items():
        image_path = os.path.join(image_folder, img_basename)
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape
        for bbox_entry in bboxes:
            bbox = bbox_entry["bbox"]
            label = bbox_entry["label"]
            x_min, y_min, x_max, y_max = bbox
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, img_w - 1)
            y_max = min(y_max + padding, img_h - 1)
            cropped = img[y_min:y_max, x_min:x_max]
            crop_filename = f"{img_basename.split('.')[0]}_{label}_{x_min}_{y_min}_{x_max}_{y_max}_padded.jpg"
            crop_path = os.path.join(crop_folder, crop_filename)
            cv2.imwrite(crop_path, cropped)
    return missing_bboxes
