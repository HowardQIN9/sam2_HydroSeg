import os
import re
import cv2
import json
import numpy as np
from datetime import datetime, timedelta

def process_masks_with_padding(missing_bbox_file, mask_json_output, output_dir, kernel_size=5, max_lookback_days=4):
    """
    按时间顺序处理missing bbox，从历史mask中padding并缓存
    
    Args:
        missing_bbox_file (str): 包含缺失bbox信息的JSON文件
        mask_json_output (str): 包含历史mask信息的JSON文件
        output_dir (str): 输出目录，保存处理后的mask
        kernel_size (int): 膨胀mask时的卷积核大小
        max_lookback_days (int): 最大回溯天数
    """
    print(f"Starting mask processing with padding. Kernel size: {kernel_size}, Max lookback days: {max_lookback_days}")
    
    missing_bbox_file = str(missing_bbox_file)
    mask_json_output = str(mask_json_output)
    output_dir = str(output_dir)
    
    print(f"Missing bbox file: {missing_bbox_file}")
    print(f"Mask JSON file: {mask_json_output}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    try:
        with open(missing_bbox_file, 'r') as f:
            missing_bbox_data = json.load(f)
        print(f"Successfully loaded missing bbox data.")
    except Exception as e:
        print(f"Error loading missing bbox file: {e}")
        return
    
    try:
        with open(mask_json_output, 'r') as f:
            mask_data = json.load(f)
        print(f"Successfully loaded mask data. Found {len(mask_data)} entries.")
    except Exception as e:
        print(f"Error loading mask JSON file: {e}")
        return
    
    # 2. 创建辅助数据结构
    mask_cache = {}  # 格式: {date_obj_id: mask}
    missing_dates_objects = set()  # 记录所有missing的日期-对象ID组合
    
    stats = {
        "processed_items": 0,
        "masks_applied": 0,
        "cache_hits": 0,
        "json_hits": 0,
        "masks_not_found": 0
    }
    
    # 打印一个示例的missing bbox条目，帮助我们理解结构
    if missing_bbox_data and next(iter(missing_bbox_data.values())):
        print(f"Sample missing bbox entry: {json.dumps(next(iter(missing_bbox_data.values()))[0], indent=2)}")
    
    # 3. 记录所有missing的日期-对象ID
    for date_key in missing_bbox_data:
        # 从键中提取日期
        date_only = extract_date_part(date_key)
        
        for bbox_info in missing_bbox_data[date_key]:
            obj_id = bbox_info['label']
            
            # 尝试从文件名中提取时间
            time_str = extract_time_from_bbox_info(bbox_info, date_key)
            
            missing_dates_objects.add(f"{date_only}_{obj_id}")
            print(f"Added missing date-object: {date_only}_{obj_id}, time: {time_str}")
    
    print(f"Found {len(missing_dates_objects)} missing date-object pairs")
    
    # 4. 按日期排序处理
    date_list = sorted(missing_bbox_data.keys())
    print(f"Found {len(date_list)} dates to process in missing bbox file")
    
    for date_key in date_list:
        # 提取纯日期部分
        date_only = extract_date_part(date_key)
        print(f"Processing date: {date_only} (from {date_key})")
        
        bbox_list = missing_bbox_data[date_key]
        
        for bbox_info in bbox_list:
            obj_id = bbox_info['label']
            
            # 从bbox_info或文件名中提取时间
            time_str = extract_time_from_bbox_info(bbox_info, date_key)
            
            print(f"Processing object ID: {obj_id} for date {date_only} time {time_str}")
            stats["processed_items"] += 1
            
            # 5. 查找前一天的mask
            mask = None
            
            # 回溯查找历史mask
            for days_back in range(1, max_lookback_days + 1):
                prev_date = get_previous_date(date_only, days_back)
                prev_key = f"{prev_date}_{obj_id}"
                
                # 5.1 先查找cache
                if prev_key in mask_cache:
                    mask = mask_cache[prev_key]
                    print(f"Using cached mask for {obj_id} from {prev_date} (days back: {days_back})")
                    stats["cache_hits"] += 1
                    break
                
                # 5.2 如果前一天不在missing box中，从mask json查找
                elif prev_key not in missing_dates_objects:
                    mask = get_mask_from_json(mask_data, obj_id, prev_date)
                    if mask is not None:
                        print(f"Found mask in JSON for {obj_id} from {prev_date} (days back: {days_back})")
                        stats["json_hits"] += 1
                        break
                
                # 5.3 如果前一天在missing box中或未找到mask，继续查找更早日期
                else:
                    print(f"Date {prev_date} for object {obj_id} is in missing list, trying earlier date")
            
            # 6. 处理找到的mask
            if mask is not None:
                padded_mask = pad_mask(mask, kernel_size)
                
                # 7. 保存处理后的mask
                output_path = os.path.join(output_dir, f"image_{date_only}_{time_str}_{obj_id}_mask.png")
                print(f"Saving padded mask to: {output_path}")
                cv2.imwrite(output_path, padded_mask)
                stats["masks_applied"] += 1
                
                # 8. 将处理后的mask添加到cache
                current_cache_key = f"{date_only}_{obj_id}"
                mask_cache[current_cache_key] = padded_mask
                print(f"Added mask for {current_cache_key} to cache")
            else:
                print(f"ERROR: No suitable mask found for {obj_id} within {max_lookback_days} days before {date_only}")
                stats["masks_not_found"] += 1
    
    print(f"Processing complete. Stats: {stats}")
    return stats

def extract_date_part(date_key):
    """从日期键中提取YYYYMMDD格式的日期"""
    # 如果是纯8位数字，直接返回
    if re.match(r'^\d{8}$', date_key):
        return date_key
    
    # 尝试从image_YYYYMMDD_HHMMSS.jpg格式中提取
    match = re.search(r'image_(\d{8})_', date_key)
    if match:
        return match.group(1)
    
    # 如果是其他格式，尝试提取任意8位数字序列
    match = re.search(r'(\d{8})', date_key)
    if match:
        return match.group(1)
    
    print(f"WARNING: Could not extract date from '{date_key}'. Using as is.")
    return date_key

def extract_time_from_bbox_info(bbox_info, filename_key):
    """从bbox信息或文件名中提取时间信息"""
    # 首先尝试直接从bbox_info中获取时间
    if 'time' in bbox_info:
        return bbox_info['time']
    
    if 'timestamp' in bbox_info:
        return bbox_info['timestamp']
    
    # 如果bbox_info中没有时间信息，尝试从文件名中提取
    # 假设格式为image_YYYYMMDD_HHMMSS.jpg
    match = re.search(r'_\d{8}_(\d{6})', filename_key)
    if match:
        return match.group(1)
    
    # 如果有原始文件名字段
    for field in ['filename', 'image_file', 'file']:
        if field in bbox_info and isinstance(bbox_info[field], str):
            match = re.search(r'_\d{8}_(\d{6})', bbox_info[field])
            if match:
                return match.group(1)
    
    # 如果都找不到，打印警告并返回默认值
    print(f"WARNING: Could not find time information for {bbox_info.get('label', 'unknown')}. Using default.")
    return "000000"

def get_previous_date(date_str, days=1):
    """获取前 `days` 天的日期"""
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        prev_date_obj = date_obj - timedelta(days=days)
        return prev_date_obj.strftime("%Y%m%d")
    except ValueError as e:
        print(f"ERROR in get_previous_date: {e}")
        print(f"Input date_str was: '{date_str}'")
        raise

def get_mask_from_json(mask_data, obj_id, date_str):
    """从 JSON 数据中获取 mask 路径并加载 mask 图像"""
    # 匹配日期对应的所有图片
    possible_keys = [key for key in mask_data.keys() if key.startswith(f"image_{date_str}_")]

    if not possible_keys:
        print(f"No entry for {date_str} in mask_data")
        return None

    for img_key in possible_keys:
        mask_list = mask_data[img_key]
        
        for entry in mask_list:
            if entry.get("label") == obj_id:
                mask_path = entry.get("mask_path")
                if mask_path and os.path.exists(mask_path):
                    print(f"Loading mask from: {mask_path}")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        return mask
                    else:
                        print(f"Failed to load mask image from {mask_path}")

    print(f"No mask found for Object ID {obj_id} in {date_str}")
    return None

def pad_mask(mask, kernel_size=5):
    """对 mask 进行膨胀操作"""
    print(f"Padding mask with kernel size {kernel_size}")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    padded_mask = cv2.dilate(mask, kernel, iterations=1)
    print(f"Original mask shape: {mask.shape}, Padded mask shape: {padded_mask.shape}")
    return padded_mask

if __name__ == "__main__":
    missing_bbox_file = "path/to/missing_bboxes.json"
    mask_json_output = "path/to/previous_day_masks.json"
    output_dir = "path/to/output"
    
    stats = process_masks_with_padding(
        missing_bbox_file=missing_bbox_file,
        mask_json_output=mask_json_output,
        output_dir=output_dir,
        kernel_size=5,
        max_lookback_days=4
    )