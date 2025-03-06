import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

def run_sam_with_prompts(prompts_dict, sam_checkpoint, model_type="vit_h",
                         image_root="", output_dir="masks"):
    """
    Runs SAM inference for each image + point prompt in prompts_dict.

    Args:
        prompts_dict (dict): A dict mapping "filename.jpg" -> 
                             {"points": [[x,y], ...], "labels": [1, ...], "mask_names": ["name1", ...]}
        sam_checkpoint (str): Path to the SAM weights file.
        model_type (str): The SAM model type (e.g., "vit_h", "vit_l", "vit_b").
        image_root (str): Directory that contains the actual image files.
        output_dir (str): Directory in which to save mask PNGs.
    """
    # 1) Initialize the SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise fallback to CPU
    predictor = SamPredictor(sam)

    # 2) Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # 3) Loop over each image in prompts_dict
    for img_basename, image_data in prompts_dict.items():
        # Ensure data format is correct
        if not isinstance(image_data, dict):
            print(f"Skipping {img_basename}: Data format incorrect")
            continue

        # Get points, labels, and mask names
        point_coords = image_data.get("points", [])
        labels = image_data.get("labels", [])
        mask_names = image_data.get("mask_names", [])

        if not (len(point_coords) == len(labels) == len(mask_names)):
            print(f"Warning: {img_basename} has mismatched data lengths. Skipping...")
            continue

        # Build the full image path
        image_path = os.path.join(image_root, img_basename)
        
        # Read the image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Failed to read image: {image_path}, skipping...")
            continue

        # Convert BGR (OpenCV) to RGB for SAM
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Set the image on the predictor
        predictor.set_image(img_rgb)

        # For each keypoint
        for i, (point, label_name) in enumerate(zip(point_coords, mask_names)):
            try:
                # Convert point to numpy array
                point_coords = np.array([point], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)  # SAM uses 1 for foreground

                # Run SAM inference
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )

                # 'masks' is a NumPy array of shape (1, H, W), we need (H, W)
                mask_array = (masks[0] * 255).astype("uint8")  # Convert boolean mask to uint8

                # Apply connected component analysis to filter out noise
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)

                if num_labels > 1:  # If there are multiple components
                    # Identify the largest connected component (excluding the background)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (label 0)

                    # Create a mask keeping only the largest component
                    filtered_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                else:
                    filtered_mask = mask_array  # If only one component, keep it as is

                # Build filename
                base_name = os.path.splitext(img_basename)[0]
                mask_filename = f"{base_name}_{label_name}_mask.png"
                save_path = os.path.join(output_dir, mask_filename)

                # Save the filtered mask
                cv2.imwrite(save_path, filtered_mask)
                print(f"Saved filtered mask: {save_path}")

            except Exception as e:
                print(f"Error processing {img_basename} - {label_name}: {str(e)}")

if __name__ == "__main__":
    # Paths customized for your environment
    points_json = "/home/zqin74/RGB/point_prompts7.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"

    # The actual folder containing your image files
    image_root_dir = "/home/zqin74/RGB/Rasp1"

    # Output folder for segmentation masks
    output_dir = "/home/zqin74/RGB/Seg_Rap1"

    # 1) Read the point prompts dictionary
    with open(points_json, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)

    # 2) Run SAM inference
    run_sam_with_prompts(
        prompts_dict=prompts_dict,
        sam_checkpoint=sam_checkpoint_path,
        model_type=sam_model_type,
        image_root=image_root_dir,
        output_dir=output_dir
    )
