import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

def run_sam_with_prompts(prompts_dict, sam_checkpoint, model_type="vit_h",
                         image_root="", output_dir="masks", mask_json_output="masks.json"):
    """
    Runs SAM inference for each image + point prompt in prompts_dict and saves masks as PNGs.
    Also stores mask file paths in JSON for later processing.
    """
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    os.makedirs(output_dir, exist_ok=True)

    mask_dict = {}  # Dictionary to store mask file paths

    for img_basename, image_data in prompts_dict.items():
        if not isinstance(image_data, dict):
            print(f"Skipping {img_basename}: Data format incorrect")
            continue

        point_coords = image_data.get("points", [])
        mask_names = image_data.get("mask_names", [])

        if not (len(point_coords) == len(mask_names)):
            print(f"Warning: {img_basename} has mismatched data lengths. Skipping...")
            continue

        image_path = os.path.join(image_root, img_basename)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Failed to read image: {image_path}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        mask_dict[img_basename] = []

        for i, (point, label_name) in enumerate(zip(point_coords, mask_names)):
            try:
                # 当前点作为正点 (label=1)，其余点作为负点 (label=0)
                positive_point = np.array([point], dtype=np.float32)
                negative_points = np.array([p for j, p in enumerate(point_coords) if j != i], dtype=np.float32)

                # 组合正负点
                if len(negative_points) > 0:
                    all_points = np.vstack((positive_point, negative_points))
                    all_labels = np.array([1] + [0] * len(negative_points), dtype=np.int32)
                else:
                    all_points = positive_point
                    all_labels = np.array([1], dtype=np.int32)

                # 进行 SAM 分割
                masks, _, _ = predictor.predict(
                    point_coords=all_points,
                    point_labels=all_labels,
                    multimask_output=False
                )

                mask_array = (masks[0] * 255).astype("uint8")

                # 仅保留最大连通区域，减少噪声
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    filtered_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                else:
                    filtered_mask = mask_array

                base_name = os.path.splitext(img_basename)[0]
                mask_filename = f"{base_name}_{label_name}_mask.png"
                save_path = os.path.join(output_dir, mask_filename)

                cv2.imwrite(save_path, filtered_mask)
                print(f"Saved filtered mask: {save_path}")

                mask_dict[img_basename].append({"mask_path": save_path, "label": label_name})

            except Exception as e:
                print(f"Error processing {img_basename} - {label_name}: {str(e)}")

    # Save mask paths to JSON for later use
    with open(mask_json_output, "w", encoding="utf-8") as f:
        json.dump(mask_dict, f, indent=4)

    print(f"Mask paths saved to: {mask_json_output}")

if __name__ == "__main__":
    points_json = "/home/zqin74/RGB/point_prompts7.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"

    image_root_dir = "/home/zqin74/RGB/Rasp1"
    output_dir = "/home/zqin74/RGB/Seg_Rap1"
    mask_json_output = "/home/zqin74/RGB/ts/masks1.json"

    with open(points_json, 'r', encoding="utf-8") as f:
        prompts_dict = json.load(f)

    run_sam_with_prompts(
        prompts_dict=prompts_dict,
        sam_checkpoint=sam_checkpoint_path,
        model_type=sam_model_type,
        image_root=image_root_dir,
        output_dir=output_dir,
        mask_json_output=mask_json_output
    )
