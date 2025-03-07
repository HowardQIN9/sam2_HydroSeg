import os
import json
import cv2
import numpy as np

def get_bounding_box(mask_path):
    """Finds the bounding box of the largest component in a binary mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask {mask_path}")
        return None

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No foreground detected

    # Convert np.int64 to Python int
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

    return [x_min, y_min, x_max, y_max]  # SAM bbox format


def generate_bboxes(mask_json_path, bbox_json_output):
    """Reads masks.json, extracts bounding boxes, and saves them to bboxes.json."""
    with open(mask_json_path, "r", encoding="utf-8") as f:
        mask_dict = json.load(f)

    bbox_dict = {}

    for img_basename, mask_list in mask_dict.items():
        bbox_dict[img_basename] = {"bboxes": [], "mask_ids": []}

        for mask_info in mask_list:
            mask_path = mask_info["mask_path"]
            label_name = mask_info["label"]

            bbox = get_bounding_box(mask_path)
            if bbox:
                bbox_dict[img_basename]["bboxes"].append({
                    "bbox": bbox,
                    "label": label_name
                })
                bbox_dict[img_basename]["mask_ids"].append(mask_path)

    with open(bbox_json_output, "w", encoding="utf-8") as f:
        json.dump(bbox_dict, f, indent=4)

    print(f"Bounding boxes saved to: {bbox_json_output}")

if __name__ == "__main__":
    mask_json_path = "/home/zqin74/RGB/ts/masks1.json"
    bbox_json_output = "/home/zqin74/RGB/ts/bboxes1.json"

    generate_bboxes(mask_json_path, bbox_json_output)
