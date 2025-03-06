import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

def load_sam_model(sam_checkpoint, model_type="vit_h"):
    """Loads the SAM model."""
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)

def process_image(image_path):
    """Reads and converts an image from BGR to RGB."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to read image: {image_path}, skipping...")
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

def run_sam_inference(predictor, img_rgb, point_coords):
    """Runs SAM inference on a given image with multiple keypoints."""
    predictor.set_image(img_rgb)
    masks_list = []
    for point in point_coords:
        point_array = np.array([point], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, _, _ = predictor.predict(
            point_coords=point_array,
            point_labels=point_labels,
            multimask_output=False
        )
        masks_list.append(masks[0])
    return masks_list

def save_mask_and_crop(img_bgr, mask, img_basename, label_name, output_dir, crop_dir, padding=10):
    """Saves the segmentation mask and crops the image based on the mask's bounding box with padding."""
    base_name = os.path.splitext(img_basename)[0]
    
    mask_filename = f"{base_name}_{label_name}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_path, (mask * 255).astype("uint8"))
    print(f"Saved mask: {mask_path}")
    
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    height, width, _ = img_bgr.shape
    
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, width)
    y2 = min(y + h + padding, height)
    
    cropped_img = img_bgr[y1:y2, x1:x2]
    crop_bbox = (x1, y1, x2, y2)
    
    crop_filename = f"{base_name}_{label_name}_crop.jpg"
    crop_path = os.path.join(crop_dir, crop_filename)
    cv2.imwrite(crop_path, cropped_img)
    print(f"Saved cropped image: {crop_path}")
    
    return crop_bbox, crop_path

def process_images_with_mapping(prompts_dict, predictor, image_root, output_dir, crop_dir, mapped_crop_dir, process_first_only=True, padding=10):
    """Processes images, maps bounding boxes to the next image, and runs SAM again on cropped images."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(mapped_crop_dir, exist_ok=True)
    
    img_list = list(prompts_dict.keys())
    for i in range(len(img_list) - 1):
        img_basename = img_list[i]
        next_img_basename = img_list[i + 1]
        
        image_path = os.path.join(image_root, img_basename)
        next_image_path = os.path.join(image_root, next_img_basename)
        
        img_bgr, img_rgb = process_image(image_path)
        next_img_bgr, next_img_rgb = process_image(next_image_path)
        if img_bgr is None or next_img_bgr is None:
            continue
        
        point_coords = prompts_dict[img_basename].get("points", [])
        mask_names = prompts_dict[img_basename].get("mask_names", [])
        
        if len(point_coords) != len(mask_names):
            print(f"Warning: {img_basename} has mismatched points and mask names. Skipping...")
            continue
        
        masks = run_sam_inference(predictor, img_rgb, point_coords)
        
        for mask, label_name in zip(masks, mask_names):
            crop_bbox, crop_path = save_mask_and_crop(img_bgr, mask, img_basename, label_name, output_dir, crop_dir, padding)
            
            x1, y1, x2, y2 = crop_bbox
            mapped_crop_img = next_img_bgr[y1:y2, x1:x2]
            mapped_crop_filename = f"{next_img_basename}_{label_name}_mapped_crop.jpg"
            mapped_crop_path = os.path.join(mapped_crop_dir, mapped_crop_filename)
            cv2.imwrite(mapped_crop_path, mapped_crop_img)
            print(f"Saved mapped cropped image: {mapped_crop_path}")
        
        if process_first_only:
            break

if __name__ == "__main__":
    points_json = "/home/zqin74/RGB/point_prompts11.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    image_root_dir = "/home/zqin74/RGB/Rasp5"
    output_dir = "/home/zqin74/RGB/Seg_Rap5"
    crop_output_dir = "/home/zqin74/RGB/Crop_Rap5"
    mapped_crop_output_dir = "/home/zqin74/RGB/Mapped_Crop_Rap5"
    
    padding_value = 10
    
    with open(points_json, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)
    
    predictor = load_sam_model(sam_checkpoint_path, sam_model_type)
    
    process_images_with_mapping(prompts_dict, predictor, image_root_dir, output_dir, crop_output_dir, mapped_crop_output_dir, process_first_only=True, padding=padding_value)
