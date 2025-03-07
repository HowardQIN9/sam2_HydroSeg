import json
import cv2
import numpy as np

def get_bounding_box(mask_path):
    """计算 mask 的 bounding box。"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    y_indices, x_indices = np.where(mask > 0)
    return [int(np.min(x_indices)), int(np.min(y_indices)), 
            int(np.max(x_indices)), int(np.max(y_indices))] if x_indices.size else None

def generate_bboxes(mask_json_path, bbox_json_output):
    """读取 mask.json，计算 bounding box 并保存到 bboxes.json。"""
    with open(mask_json_path, "r", encoding="utf-8") as f:
        mask_dict = json.load(f)

    bbox_dict = {img: [{"bbox": get_bounding_box(info["mask_path"]), "label": info["label"]} 
                        for info in masks] for img, masks in mask_dict.items()}

    with open(bbox_json_output, "w", encoding="utf-8") as f:
        json.dump(bbox_dict, f, indent=4)

    print(f"Bounding boxes saved to: {bbox_json_output}")
