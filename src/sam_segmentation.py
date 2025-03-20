import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

def run_sam(prompts_list, sam_checkpoint, model_type="vit_h", 
            image_root="", output_dir="masks", mask_json_output="masks.json"):
    """使用 SAM 模型对图像进行分割，并保存处理后的 mask。"""
    
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    os.makedirs(output_dir, exist_ok=True)
    # Create directory for mask_json_output if it doesn't exist
    json_output_dir = os.path.dirname(mask_json_output)
    if json_output_dir:  # Check if there's a directory component
        os.makedirs(json_output_dir, exist_ok=True)
    
    # Clear existing JSON file if it exists
    if os.path.exists(mask_json_output):
        # print(f"Clearing existing file: {mask_json_output}")
        open(mask_json_output, 'w').close()  # This will empty the file

    mask_dict = {}
    total_images = len(prompts_list)
    image_counter = 0

    for img_basename, image_data in prompts_list.items():

        image_counter +=1
        print(f"Processing image {image_counter}/{total_images}: {img_basename}")

        image_path = os.path.join(image_root, img_basename)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Failed to read image: {image_path}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        mask_dict[img_basename] = []

        for point, label_name in zip(image_data["point_coords"], image_data["mask_names"]):
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
                filtered_mask = mask_array

            # 形态学操作（加速去噪）
            kernel = np.ones((3, 3), np.uint8)  # 定义一个较小的5x5内核
            # 开运算（去掉小的噪声）
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            # 闭运算（填补孔洞）
            filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 再次提取最大连通区域，确保连通区域内没有其他区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 重新提取最大连通区域
                final_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            else:
                final_mask = filtered_mask

            mask_filename = f"{os.path.splitext(img_basename)[0]}_{label_name}_mask.png"
            save_path = os.path.join(output_dir, mask_filename)

            cv2.imwrite(save_path, final_mask)
            mask_dict[img_basename].append({"mask_path": save_path, "label": label_name})

    with open(mask_json_output, "w", encoding="utf-8") as f:
        json.dump(mask_dict, f, indent=4)

    print(f"Mask paths saved to: {mask_json_output}")
