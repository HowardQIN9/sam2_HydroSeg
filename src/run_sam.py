import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

import os
import cv2
import json
import numpy as np
import torch
import datetime
from segment_anything import SamPredictor, sam_model_registry

def get_bounding_box(mask):
    """计算 mask 的 bounding box（传入mask数组而非路径）"""
    y_indices, x_indices = np.where(mask > 0)
    if x_indices.size == 0:
        return None  # 空 mask，返回 None
    return [int(np.min(x_indices)), int(np.min(y_indices)), 
            int(np.max(x_indices)), int(np.max(y_indices))]

def extract_date_from_filename(filename):
    """从文件名中提取日期，支持 image_YYYYMMDD_HHMMSS.jpg 形式"""
    try:
        parts = filename.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 8:  # 识别 YYYYMMDD
                return datetime.datetime.strptime(part, "%Y%m%d")
        return None  # 没有找到日期
    except ValueError:
        return None

def run_sam(prompts_list, sam_checkpoint, model_type="vit_h", 
            image_root="", output_dir="masks", mask_json_output="masks.json",
            min_ratio=5/7, max_ratio=7/5, bad_case_json_output="bad_cases.json",
            start_date=None):  # 添加 start_date 参数
    
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    os.makedirs(output_dir, exist_ok=True)
    mask_dict = {}
    bad_cases = {}

    for img_basename, image_data in prompts_list.items():
        image_path = os.path.join(image_root, img_basename)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Failed to read image: {image_path}, skipping...")
            continue

        # 提取文件日期，判断是否进行 bad case 检测
        img_date = extract_date_from_filename(img_basename)
        if img_date is None:
            print(f"Warning: 无法解析 {img_basename} 的日期，跳过 bad case 检测，但仍然保存 mask")
        elif start_date and img_date < start_date:
            print(f"Skipping bad case check for {img_basename} (date {img_date.strftime('%Y-%m-%d')} < {start_date.strftime('%Y-%m-%d')})")
            skip_bad_case_check = True
        else:
            skip_bad_case_check = False  # 只有在 start_date 之后才做 bad case 检测

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        mask_dict[img_basename] = []  # **无论如何都要存 mask**
        bad_cases[img_basename] = []

        for point, label_name in zip(image_data["points"], image_data["mask_names"]):
            point_coords = np.array([point], dtype=np.float32)
            masks, _, _ = predictor.predict(point_coords=point_coords, 
                                            point_labels=np.array([1], dtype=np.int32),
                                            multimask_output=False)

            mask_array = (masks[0] * 255).astype("uint8")

            # 处理 mask，仅保留最大连通区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 忽略背景（label 0）
                filtered_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            else:
                filtered_mask = mask_array  # 只有一个连通区域，直接使用

            # 计算 bounding box
            bbox = get_bounding_box(filtered_mask)
            if bbox is None:
                print(f"Skipping empty mask for {img_basename} - {label_name}")
                continue

            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            # **只在 start_date 之后进行 bad case 过滤**
            if not skip_bad_case_check:
                if height == 0:  # 避免除零错误
                    print(f"Skipping degenerate mask (height=0) for {img_basename} - {label_name}")
                    continue

                aspect_ratio = width / height

                if not (min_ratio <= aspect_ratio <= max_ratio):
                    print(f"Bad case detected for {img_basename} - {label_name} (Aspect Ratio: {aspect_ratio:.2f})")
                    bad_cases[img_basename].append({
                        "bbox": bbox,
                        "label": label_name,
                        "aspect_ratio": aspect_ratio
                    })
                    continue  # 直接跳过存储 mask 文件，但仍然存到 JSON

            # 保存 mask
            mask_filename = f"{os.path.splitext(img_basename)[0]}_{label_name}_mask.png"
            save_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(save_path, filtered_mask)

            mask_dict[img_basename].append({"mask_path": save_path, "label": label_name, "bbox": bbox})

    # 保存 masks.json（包含所有 mask）
    with open(mask_json_output, "w", encoding="utf-8") as f:
        json.dump(mask_dict, f, indent=4)
    print(f"Valid masks saved to: {mask_json_output}")

    # 保存 bad_cases.json（只包含 start_date 之后的 bad case）
    with open(bad_case_json_output, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=4)
    print(f"Bad cases saved to: {bad_case_json_output}")

def crop_bad_cases(bad_case_json, image_root_dir, crop_output_dir):
    """裁剪 bad case 并存入 crop_output_dir"""
    os.makedirs(crop_output_dir, exist_ok=True)

    with open(bad_case_json, "r", encoding="utf-8") as f:
        bad_cases = json.load(f)

    for filename, file_data in bad_cases.items():
        image_path = os.path.join(image_root_dir, filename)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            continue

        for bbox_entry in file_data:
            obj_id = bbox_entry.get("label")
            bbox = bbox_entry.get("bbox")
            if bbox is None or len(bbox) != 4:
                print(f"[WARNING] Invalid bbox for {filename}: {bbox}")
                continue

            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_img = img[y_min:y_max, x_min:x_max]

            crop_filename = f"{filename[:-4]}_{obj_id}_{x_min}_{y_min}_{x_max}_{y_max}_cropped.jpg"
            crop_path = os.path.join(crop_output_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_img)
            print(f"[INFO] Saved cropped bad case: {crop_path}")

    print("[INFO] Cropping bad cases completed.")





def main():
    # 配置路径
    points_json = "/home/zqin74/RGB/point_prompts9.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    image_root_dir = "/home/zqin74/RGB/Rasp3"
    output_dir = "/home/zqin74/RGB/v2/Seg_Rap3"
    mask_json_output = "/home/zqin74/RGB/v2/masks3.json"
    bad_case_json_output = "/home/zqin74/RGB/v2/bad_case3.json"
    crop_output_dir = "/home/zqin74/RGB/v2/crop3"
    sam_model_type = "vit_h"

    # 设定检测的起始日期
    start_date = datetime.datetime(2024, 10, 26)  

    # 读取 point prompt 数据
    if not os.path.exists(points_json):
        print(f"Error: {points_json} 文件不存在")
        return

    with open(points_json, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    # 运行 SAM 分割
    run_sam(
        prompts_list=prompts_list,
        sam_checkpoint=sam_checkpoint_path,
        model_type=sam_model_type,
        image_root=image_root_dir,
        output_dir=output_dir,
        mask_json_output=mask_json_output,
        bad_case_json_output=bad_case_json_output,
        start_date=start_date
    )

    # 裁剪 bad case
    print("[INFO] Cropping bad cases from original images...")
    crop_bad_cases(bad_case_json_output, image_root_dir, crop_output_dir)

if __name__ == "__main__":
    main()

