import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

def run_sam(prompts_list, sam_checkpoint, model_type="vit_h", 
            image_root="", output_dir="masks", mask_json_output="masks.json"):
    """使用 SAM 模型对图像进行分割，并保存 mask。"""
    
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    os.makedirs(output_dir, exist_ok=True)
    mask_dict = {}

    for img_basename, image_data in prompts_list.items():
        image_path = os.path.join(image_root, img_basename)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Failed to read image: {image_path}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        mask_dict[img_basename] = []

        for point, label_name in zip(image_data["points"], image_data["mask_names"]):
            point_coords = np.array([point], dtype=np.float32)
            masks, _, _ = predictor.predict(point_coords=point_coords, 
                                            point_labels=np.array([1], dtype=np.int32),
                                            multimask_output=False)

            mask_array = (masks[0] * 255).astype("uint8")
            mask_filename = f"{os.path.splitext(img_basename)[0]}_{label_name}_mask.png"
            save_path = os.path.join(output_dir, mask_filename)

            cv2.imwrite(save_path, mask_array)
            mask_dict[img_basename].append({"mask_path": save_path, "label": label_name})

    with open(mask_json_output, "w", encoding="utf-8") as f:
        json.dump(mask_dict, f, indent=4)

    print(f"Mask paths saved to: {mask_json_output}")
