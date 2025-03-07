import os
import re
import cv2
import numpy as np

def parse_mask_filename(filename):
    """
    Extracts date, object ID from the mask filename.
    Example: image_20241111_130006_T12F7_1648_187_1899_364_mask.png
    Returns: (shortened_filename, crop_bbox)
    """
    pattern = r"image_(\d{8})_(\d{6})_(T\d+[A-Z]\d+)_\d+_\d+_\d+_\d+_mask\.png"
    match = re.match(pattern, filename)
    if match:
        date_str, time_str, obj_id = match.groups()
        shortened_filename = f"image_{date_str}_{time_str}_{obj_id}_mask.png"
        return shortened_filename
    else:
        raise ValueError(f"[ERROR] Filename format is incorrect: {filename}")

def project_mask_to_original(small_mask_path, original_image_shape, output_folder):
    """
    Projects the small mask back to the original mask size and saves it.

    Parameters:
    - small_mask_path: Path to the small mask image
    - original_image_shape: (H, W) of the original full-sized image
    - output_folder: Directory to save projected masks
    
    Returns:
    - Path to the saved projected mask
    """
    try:
        # Extract shortened filename
        filename = os.path.basename(small_mask_path)
        shortened_filename = parse_mask_filename(filename)

        # Extract crop bounding box from the full filename
        bbox_pattern = r"_(\d+)_(\d+)_(\d+)_(\d+)_mask\.png"
        bbox_match = re.search(bbox_pattern, filename)
        if not bbox_match:
            raise ValueError(f"[ERROR] Bounding box not found in filename: {filename}")

        x_min, y_min, x_max, y_max = map(int, bbox_match.groups())

        # Load the small mask
        small_mask = cv2.imread(small_mask_path, cv2.IMREAD_GRAYSCALE)
        if small_mask is None:
            raise FileNotFoundError(f"[ERROR] Failed to load mask: {small_mask_path}")

        # Ensure mask dimensions match the bounding box
        mask_h, mask_w = small_mask.shape
        expected_w, expected_h = x_max - x_min, y_max - y_min
        if (mask_w, mask_h) != (expected_w, expected_h):
            print(f"[WARNING] Mask size {mask_w}x{mask_h} does not match expected {expected_w}x{expected_h}, resizing.")
            small_mask = cv2.resize(small_mask, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)

        # Create a blank mask with the same size as the original image
        full_mask = np.zeros(original_image_shape, dtype=np.uint8)

        # Place the small mask into the correct position in the full-sized mask
        full_mask[y_min:y_max, x_min:x_max] = small_mask

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        projected_mask_path = os.path.join(output_folder, shortened_filename)

        # Save the projected mask
        cv2.imwrite(projected_mask_path, full_mask)
        print(f"[SUCCESS] Projected mask saved: {projected_mask_path}")

        return projected_mask_path
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def process_mask_folder(mask_folder, original_image_shape, output_folder):
    """
    Processes all mask images in a given folder and projects them back to the original size.

    Parameters:
    - mask_folder: Path to the folder containing small mask images
    - original_image_shape: (H, W) of the original full-sized image
    - output_folder: Path to save projected masks
    
    Returns:
    - None (saves projected masks)
    """
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith("_mask.png")]

    if not mask_files:
        print("[WARNING] No mask files found in the folder.")
        return

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        project_mask_to_original(mask_path, original_image_shape, output_folder)

# Example usage
if __name__ == "__main__":
    mask_folder = "/home/zqin74/RGB/IOU"  # 输入 mask 文件夹路径
    output_folder = "/home/zqin74/RGB/projected_masks"  # 结果存放路径
    original_image_shape = (1080, 1920)  # 设定原始图像尺寸

    process_mask_folder(mask_folder, original_image_shape, output_folder)
