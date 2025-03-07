import json

def filter_bad_cases(bbox_json_path, bad_case_json_output, min_ratio=5/7, max_ratio=7/5):
    """Filters bounding boxes based on aspect ratio and saves results to bad_case.json."""
    
    # Load bbox data from JSON
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    bad_cases = {}

    for img, bboxes in bbox_data.items():
        bad_cases[img] = []
        
        for b in bboxes:
            # Ensure bbox values are converted to integers
            x_min, y_min, x_max, y_max = map(int, b["bbox"])  # Convert to integers

            width = x_max - x_min
            height = y_max - y_min
            
            if height == 0:  # Avoid division by zero
                continue  

            aspect_ratio = width / height

            if not (min_ratio <= aspect_ratio <= max_ratio):  # Ensure proper comparison
                bad_cases[img].append({
                    "bbox": [x_min, y_min, x_max, y_max], 
                    "label": b["label"], 
                    "aspect_ratio": aspect_ratio
                })

    # Save bad cases
    with open(bad_case_json_output, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=4)

    print(f"Bad cases saved to: {bad_case_json_output}")
