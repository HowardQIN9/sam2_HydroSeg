import json
from datetime import datetime

def extract_date_from_filename(filename):
    """Extracts the date from the image filename assuming format: image_YYYYMMDD_HHMMSS.jpg"""
    try:
        date_part = filename.split("_")[1]  # Extract YYYYMMDD part
        return datetime.strptime(date_part, "%Y%m%d")  # Convert to datetime object
    except (IndexError, ValueError):
        return None  # Return None if the date extraction fails

def filter_bad_cases(bbox_json_path, bad_case_json_output, start_date, min_ratio=10/14, max_ratio=14/10):
    """Filters bounding boxes based on aspect ratio and saves results to bad_case.json."""
    
    # Load bbox data from JSON
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    bad_cases = {}
    start_date = datetime.strptime(start_date, "%Y-%m-%d")  # Convert start_date to datetime object

    for img, bboxes in bbox_data.items():
        img_date = extract_date_from_filename(img)  # Extract date from filename

        if img_date is None or img_date < start_date:  # Skip if date is invalid or before start_date
            continue  

        bad_cases[img] = []

        for b in bboxes:
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

    # Remove empty entries (no bad cases)
    bad_cases = {img: cases for img, cases in bad_cases.items() if cases}

    # Save bad cases
    with open(bad_case_json_output, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=4)

    print(f"Bad cases saved to: {bad_case_json_output}")

def main():
    bbox_json_path = "input.json"  # Replace with actual path
    bad_case_json_output = "bad_case.json"
    start_date = "2024-10-18"  # Set filtering start date

    filter_bad_cases(bbox_json_path, bad_case_json_output, start_date)

if __name__ == "__main__":
    main()
