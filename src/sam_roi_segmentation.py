import os
import re
import cv2
import json
import numpy as np
import torch
from datetime import datetime, timedelta
# from segment_anything import SamPredictor, sam_model_registry

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
            # print(f"[DEBUG] Checking {filename}: {file_data}")

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



# def run_sam_on_cropped_image(small_image_path, bbox_prompt, sam_model):
#     img = cv2.imread(small_image_path)
#     if img is None:
#         print(f"[ERROR] Failed to read: {small_image_path}")
#         return None
    
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     predictor = SamPredictor(sam_model)
#     predictor.set_image(img_rgb)

#     bbox_prompt = np.array(bbox_prompt, dtype=np.float32).reshape(1, 4)

#     masks, _, _ = predictor.predict(box=bbox_prompt)
#     return masks[0]

# def run_sam_on_rois(small_image_folder, bbox_json_path, sam_checkpoint, model_type="vit_h"):
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

#     bbox_cache = {}  # 用于缓存 {日期: {obj_id: bbox}}

#     for filename in sorted(os.listdir(small_image_folder)):  # 确保按照日期顺序处理
#         if not filename.endswith("_padded.jpg"):
#             continue

#         date_str, obj_id, crop_bbox = parse_small_image_filename(filename)
#         if not date_str or not obj_id:
#             print(f"[WARNING] Failed to parse filename: {filename}")
#             continue

#         prev_date = get_previous_date(date_str)
#         print(f"\n[INFO] Processing: {filename}, Object ID: {obj_id}, Date: {date_str} -> {prev_date}")

#         # **优先从缓存找 bbox**
#         if prev_date in bbox_cache and obj_id in bbox_cache[prev_date]:  
#             prev_bbox = bbox_cache[prev_date][obj_id]  # 复用相邻日期的 bbox
#             print(f"[CACHE HIT] Using cached bbox for {obj_id} from {prev_date}: {prev_bbox}")
#         else:
#             prev_bbox = get_previous_bbox(bbox_json_path, prev_date, obj_id)  
#             if prev_bbox is None:
#                 print(f"[WARNING] No BBox found for previous day: {obj_id}")
#                 continue  # 没有 bbox 跳过

#         new_bbox = map_bbox_to_cropped_image(prev_bbox, crop_bbox)
#         print(f"[INFO] Previous BBox: {prev_bbox} -> Cropped Image BBox: {new_bbox}")

#         small_image_path = os.path.join(small_image_folder, filename)
#         mask = run_sam_on_cropped_image(small_image_path, new_bbox, sam)

#         if mask is not None:
#             # **确保 mask 转换为 uint8**
#             mask = (mask * 255).astype(np.uint8)

#             # **后处理：形态学操作和最大连通区域选择**
#             kernel = np.ones((3, 3), np.uint8)
#             mask = cv2.erode(mask, kernel, iterations=1)
#             mask = cv2.dilate(mask, kernel, iterations=1)

#             num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#             if num_labels > 1:
#                 largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#                 filtered_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
#             else:
#                 filtered_mask = mask

#             mask_save_path = small_image_path.replace("_padded.jpg", "_mask.png")
#             cv2.imwrite(mask_save_path, filtered_mask)
#             print(f"[SUCCESS] Saved segmentation mask: {mask_save_path}")

#             # **缓存最新的 bbox**
#             if date_str not in bbox_cache:
#                 bbox_cache[date_str] = {}
#             bbox_cache[date_str][obj_id] = new_bbox
#             print(f"[CACHE UPDATE] Stored bbox for {obj_id} on {date_str}: {new_bbox}")


def load_previous_mask(prev_mask_path):
    """加载前一天的 mask"""
    if os.path.exists(prev_mask_path):
        return cv2.imread(prev_mask_path, cv2.IMREAD_GRAYSCALE)
    return None

def pad_mask(mask, kernel_size=5):
    """对 mask 进行 padding 以适应目标的增长"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    padded_mask = cv2.dilate(mask, kernel, iterations=1)
    return padded_mask

def adjust_mask_to_crop_bbox(prev_mask, prev_crop_bbox, curr_crop_bbox, original_size):
    """
    调整前一天的 mask 以适应今天的裁剪区域。
    需要先把前一天的 mask 变换到原图坐标，再裁剪到今天的 crop_bbox。
    """
    px_min, py_min, px_max, py_max = prev_crop_bbox
    cx_min, cy_min, cx_max, cy_max = curr_crop_bbox
    img_w, img_h = original_size

    # 计算偏移量
    offset_x, offset_y = px_min - cx_min, py_min - cy_min

    # 计算新的 mask 位置
    h, w = prev_mask.shape
    new_mask = np.zeros((cy_max - cy_min, cx_max - cx_min), dtype=np.uint8)

    # 确保偏移量在合理范围
    x_start = max(0, offset_x)
    y_start = max(0, offset_y)
    x_end = min(cx_max - cx_min, offset_x + w)
    y_end = min(cy_max - cy_min, offset_y + h)

    # 计算 mask 的目标区域
    prev_x_start = max(0, -offset_x)
    prev_y_start = max(0, -offset_y)
    prev_x_end = prev_x_start + (x_end - x_start)
    prev_y_end = prev_y_start + (y_end - y_start)

    # 将前一天的 mask 复制到新位置
    new_mask[y_start:y_end, x_start:x_end] = prev_mask[prev_y_start:prev_y_end, prev_x_start:prev_x_end]

    return new_mask

def process_masks_with_padding(small_image_folder, prev_mask_folder, original_size, kernel_size=5):
    """
    使用前一天的 mask 进行 padding 以生成新 mask
    """
    for filename in sorted(os.listdir(small_image_folder)):
        if not filename.endswith("_padded.jpg"):
            continue
        
        date_str, obj_id, crop_bbox = parse_small_image_filename(filename)
        prev_date = get_previous_date(date_str)

        small_image_path = os.path.join(small_image_folder, filename)
        prev_mask_path = os.path.join(prev_mask_folder, f"image_{prev_date}_{obj_id}_mask.png")
        new_mask_path = small_image_path.replace("_padded.jpg", "_mask.png")

        prev_mask = load_previous_mask(prev_mask_path)

        if prev_mask is not None:
            print(f"[INFO] Using previous mask: {prev_mask_path}")

            # 对 mask 进行 padding 扩展
            padded_mask = pad_mask(prev_mask, kernel_size=kernel_size)

            # 重新映射到当前裁剪区域
            new_mask = adjust_mask_to_crop_bbox(padded_mask, prev_crop_bbox, crop_bbox, original_size)

            # 保存新的 mask
            cv2.imwrite(new_mask_path, new_mask)
            print(f"[SUCCESS] Saved adjusted mask: {new_mask_path}")
        else:
            print(f"[WARNING] No previous mask found for {obj_id} on {prev_date}, fallback to SAM.")

            # 在这里你可以调用 run_sam_on_cropped_image 作为备用方案
