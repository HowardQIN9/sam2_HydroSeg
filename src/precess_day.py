import os
import re
import cv2
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict

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

def extract_date(filename):
    """从文件名提取日期"""
    match = re.match(r"^image_(\d{8})_\d{6}\.jpg$", filename)
    return match.group(1) if match else None

def extract_datetime(filename):
    """从文件名提取日期和时间"""
    match = re.match(r"^image_(\d{8}_\d{6})\.jpg$", filename)
    return match.group(1) if match else None

class ReferenceManager:
    """管理参考图像和标注的类"""
    def __init__(self, reference_file="reference_data.pkl"):
        self.reference_file = reference_file
        self.label_to_reference = {}  # {label: {"date": date_str, "bbox": bbox}}
        self.load_references()
    
    def load_references(self):
        """加载参考数据"""
        if os.path.exists(self.reference_file):
            try:
                with open(self.reference_file, "rb") as f:
                    self.label_to_reference = pickle.load(f)
            except Exception as e:
                print(f"加载参考数据失败: {e}")
    
    def save_references(self):
        """保存参考数据"""
        with open(self.reference_file, "wb") as f:
            pickle.dump(self.label_to_reference, f)
    
    def get_reference_for_label(self, label):
        """获取标签的参考数据"""
        return self.label_to_reference.get(label, None)
    
    def update_reference_for_label(self, label, date_str, bbox):
        """更新标签的参考数据"""
        self.label_to_reference[label] = {"date": date_str, "bbox": bbox}
        self.save_references()
    
    def update_references_from_corrected(self, date_str, corrected_data):
        """从已纠正的数据更新参考"""
        for img_basename, bboxes in corrected_data.items():
            if extract_date(img_basename) == date_str:
                for bbox_entry in bboxes:
                    label = bbox_entry.get("label")
                    if label:
                        self.update_reference_for_label(label, date_str, bbox_entry["bbox"])

def gather_bboxes_for_day(bbox_data, date_yyyymmdd):
    """收集某一天的所有bbox信息，返回{image_name: {label: bbox}}形式的字典"""
    image_label_bboxes = {}
    pattern_day = f"^image_{date_yyyymmdd}_\\d{{6}}\\.jpg$"

    for filename, file_data in bbox_data.items():
        if re.match(pattern_day, filename):
            image_label_bboxes[filename] = {}
            bbox_entries = file_data if isinstance(file_data, list) else file_data.get("bboxes", [])
            for bbox_entry in bbox_entries:
                label = bbox_entry.get("label", None)
                if label:
                    image_label_bboxes[filename][label] = bbox_entry["bbox"]
    
    return image_label_bboxes

def process_missing_bboxes(bad_case_json, bbox_json_path, image_folder, crop_folder, json_output_path, 
                          corrected_data_path=None, iou_threshold=0.8, padding=10):
    """
    处理缺失的边界框
    
    参数:
    bad_case_json: bad case数据的JSON文件路径
    bbox_json_path: 边界框数据的JSON文件路径
    image_folder: 图像文件夹路径
    crop_folder: 裁剪图像的保存文件夹
    json_output_path: 输出JSON文件路径
    corrected_data_path: 已纠正数据的JSON文件路径（如果有）
    iou_threshold: IoU阈值，默认0.8
    padding: 裁剪图像时的填充像素，默认10
    """
    os.makedirs(crop_folder, exist_ok=True)
    
    # 初始化参考管理器
    reference_manager = ReferenceManager()
    
    # 加载数据
    with open(bad_case_json, "r", encoding="utf-8") as f:
        bad_cases = json.load(f)
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    
    # 加载已纠正的数据（如果有）
    corrected_data = {}
    if corrected_data_path and os.path.exists(corrected_data_path):
        with open(corrected_data_path, "r", encoding="utf-8") as f:
            corrected_data = json.load(f)
            # 更新参考数据
            for date_str in set(extract_date(img) for img in corrected_data.keys() if extract_date(img)):
                reference_manager.update_references_from_corrected(date_str, corrected_data)
    
    # 处理缺失边界框
    missing_bboxes = {}
    
    # 按日期组织坏样本
    date_to_bad_cases = defaultdict(dict)
    for img_basename, bboxes in bad_cases.items():
        date_str = extract_date(img_basename)
        if date_str:
            date_to_bad_cases[date_str][img_basename] = bboxes
    
    # 按日期顺序处理
    for date_str in sorted(date_to_bad_cases.keys()):
        for img_basename, current_bboxes in date_to_bad_cases[date_str].items():
            img_datetime = extract_datetime(img_basename)
            if not img_datetime:
                continue
                
            for cur_bbox_entry in current_bboxes:
                c_label = cur_bbox_entry.get("label", None)
                c_bbox = cur_bbox_entry["bbox"]
                
                if not c_label:
                    # 无标签，直接添加为缺失
                    missing_bboxes.setdefault(img_basename, []).append({
                        "bbox": c_bbox,
                        "label": "Unknown",
                        "image_datetime": img_datetime
                    })
                    continue
                
                # 获取参考数据
                reference = reference_manager.get_reference_for_label(c_label)
                
                if reference:
                    # 有参考数据，计算IoU
                    ref_bbox = reference["bbox"]
                    iou = compute_iou(c_bbox, ref_bbox)
                    
                    if iou < iou_threshold:
                        # IoU低于阈值，判断为缺失
                        missing_bboxes.setdefault(img_basename, []).append({
                            "bbox": c_bbox,
                            "label": c_label,
                            "image_datetime": img_datetime,
                            "reference_date": reference["date"]
                        })
                else:
                    # 没有参考数据，将当前数据设为参考
                    reference_manager.update_reference_for_label(c_label, date_str, c_bbox)
    
    # 保存缺失边界框信息
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(missing_bboxes, f, indent=4)
    
    # 裁剪保存图像
    for img_basename, bboxes in missing_bboxes.items():
        image_path = os.path.join(image_folder, img_basename)
        if not os.path.exists(image_path):
            continue
            
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        img_h, img_w, _ = img.shape
        
        for bbox_entry in bboxes:
            bbox = bbox_entry["bbox"]
            label = bbox_entry["label"]
            image_datetime = bbox_entry["image_datetime"]
            ref_date = bbox_entry.get("reference_date", "no_ref")
            
            x_min, y_min, x_max, y_max = bbox
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, img_w - 1)
            y_max = min(y_max + padding, img_h - 1)
            
            cropped = img[y_min:y_max, x_min:x_max]
            crop_filename = f"{image_datetime}_{label}_ref{ref_date}_{x_min}_{y_min}_{x_max}_{y_max}_padded.jpg"
            crop_path = os.path.join(crop_folder, crop_filename)
            cv2.imwrite(crop_path, cropped)
    
    return missing_bboxes

def update_references(corrected_data_path, reference_manager=None):
    """
    从已纠正的数据更新参考
    
    参数:
    corrected_data_path: 已纠正数据的JSON文件路径
    reference_manager: 参考管理器实例（可选）
    
    返回:
    reference_manager: 更新后的参考管理器
    """
    if reference_manager is None:
        reference_manager = ReferenceManager()
        
    with open(corrected_data_path, "r", encoding="utf-8") as f:
        corrected_data = json.load(f)
        
    for img_basename, bboxes in corrected_data.items():
        date_str = extract_date(img_basename)
        if not date_str:
            continue
            
        for bbox_entry in bboxes:
            label = bbox_entry.get("label")
            if label:
                reference_manager.update_reference_for_label(label, date_str, bbox_entry["bbox"])
                
    return reference_manager