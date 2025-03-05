import json
import os
import re

def extract_date(image_name):
    """
    Extract date from image filename in format: image_YYYYMMDD_
    Returns an integer YYYYMMDD for sorting.
    """
    match = re.search(r"image_(\d{8})_", image_name)
    return int(match.group(1)) if match else float('inf')  # Use 'inf' for unmatched cases (to push them to the end)

def label_studio_to_sam(json_path, output_json):
    """
    Convert Label Studio keypoints JSON to SAM point prompts with images sorted by date.
    
    Output format:
    {
        "image_20241112_130006.jpg": {
            "points": [[x1, y1], [x2, y2], ...],
            "labels": [1, 1, ...],  # SAM uses 1 (foreground)
            "mask_names": ["lettuce_1", "lettuce_2", ...]  # Associated mask names
        }
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sam_points = {}

    for task in data:
        img_full_path = task["data"].get("image", "")
        img_basename = os.path.basename(img_full_path)  # Extract filename
        
        # Normalize filename by removing any prefix before "image_"
        if "-" in img_basename:
            parts = img_basename.split("-", 1)
            if len(parts) > 1 and parts[1].startswith("image_"):
                img_basename = parts[1]

        for annotation in task.get("annotations", []):
            for result_item in annotation.get("result", []):
                if result_item["type"] == "keypointlabels":
                    orig_w = result_item.get("original_width", 1)  # Avoid division by zero
                    orig_h = result_item.get("original_height", 1)

                    x_pct = result_item["value"]["x"]
                    y_pct = result_item["value"]["y"]

                    # Convert from percentage to absolute pixel coordinates (rounded integers)
                    x_abs = int(round((x_pct / 100.0) * orig_w))
                    y_abs = int(round((y_pct / 100.0) * orig_h))

                    # Extract mask name (from keypoint label value)
                    mask_name = result_item["value"].get("keypointlabels", [""])[0]  # Assume first label

                    if img_basename not in sam_points:
                        sam_points[img_basename] = {"points": [], "labels": [], "mask_names": []}

                    sam_points[img_basename]["points"].append([x_abs, y_abs])
                    sam_points[img_basename]["labels"].append(1)  # SAM points are typically foreground (1)
                    sam_points[img_basename]["mask_names"].append(mask_name)

    # Sort images based on extracted date from filename
    sorted_sam_points = {k: sam_points[k] for k in sorted(sam_points.keys(), key=extract_date)}

    # Save to output JSON file
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(sorted_sam_points, fp, indent=2, ensure_ascii=False)

    print(f"Converted JSON saved to: {output_json}")

# Run the conversion
if __name__ == "__main__":
    input_json_file = "/home/zqin74/RGB/project-11.json"  # Update with your file path
    output_json_file = "/home/zqin74/RGB/point_prompts11.json"

    label_studio_to_sam(input_json_file, output_json_file)
