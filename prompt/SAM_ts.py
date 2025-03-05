import os
import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

def compute_bounding_box(mask, padding=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x1, y1, x2, y2 = float('inf'), float('inf'), 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1 = min(x1, x - padding), min(y1, y - padding)
        x2, y2 = max(x2, x + w + padding), max(y2, y + h + padding)

    return max(0, x1), max(0, y1), min(mask.shape[1], x2), min(mask.shape[0], y2)

def crop_image(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]

def run_sam_inference_first_image(predictor, img, keypoints, mask_names, output_dir, cropped_output_dir):
    print("Starting First image...")
    if img is None:
        print("Error: Image is None")
        return {}

    predictor.set_image(img)
    cropped_images = {}

    for point, label_name in zip(keypoints, mask_names):
        point_coords_array = np.array([point], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, logits = predictor.predict(point_coords=point_coords_array, point_labels=point_labels, multimask_output=False)
        if masks is None or len(masks) == 0:
            print(f"Warning: No mask generated for {label_name}")
            continue

        mask_array = (masks[0] * 255).astype("uint8")
        bbox = compute_bounding_box(mask_array)
        if bbox:
            cropped_img = crop_image(img, bbox)
            cropped_path = os.path.join(cropped_output_dir, f"{label_name}_cropped.png")
            cv2.imwrite(cropped_path, cropped_img)
            cropped_images[label_name] = cropped_path

    return cropped_images

def run_sam_inference_with_cropped(predictor, cropped_img_path, keypoint, mask_name, output_dir, cropped_output_dir):
    img_bgr = cv2.imread(cropped_img_path)
    if img_bgr is None:
        print(f"Failed to read cropped image: {cropped_img_path}, skipping...")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    try:
        point_coords_array = np.array([keypoint], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, logits = predictor.predict(point_coords=point_coords_array, point_labels=point_labels, multimask_output=False)

        if masks is None or len(masks) == 0:
            print(f"Warning: No mask generated for {mask_name}")
            return None

        mask_array = (masks[0] * 255).astype("uint8")

        mask_filename = f"{mask_name}_mask.png"
        save_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(save_path, mask_array)

        bbox = compute_bounding_box(mask_array)
        if bbox:
            cropped_img = crop_image(img_rgb, bbox)
            cropped_filename = f"{mask_name}_cropped.png"
            cropped_path = os.path.join(cropped_output_dir, cropped_filename)
            cv2.imwrite(cropped_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            return cropped_path

    except Exception as e:
        print(f"Error processing cropped image {mask_name}: {str(e)}")

    return None

def process_images_in_sequence(prompts_dict, sam_checkpoint, model_type, image_root, output_dir, cropped_output_dir):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_output_dir, exist_ok=True)

    cropped_images = {}
    print("Loading SAM..")

    first_image_name = "image_20241016_130006.jpg"
    
    # 先处理指定的第一张图片
    if first_image_name in prompts_dict:
        first_data = prompts_dict[first_image_name]
        first_img_path = os.path.join(image_root, first_image_name)
        first_img = cv2.imread(first_img_path)

        if first_img is None:
            print(f"Error: Unable to read first image {first_img_path}")
        else:
            cropped_images = run_sam_inference_first_image(predictor, first_img, first_data["points"], first_data["mask_names"], output_dir, cropped_output_dir)

    # 处理剩下的图片
    for img_basename, data in prompts_dict.items():
        if img_basename == first_image_name:
            continue  # 跳过已处理的第一张图片

        img_path = os.path.join(image_root, img_basename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image {img_path}")
            continue

        for mask_name, cropped_path in cropped_images.items():
            if cropped_path:
                new_cropped = run_sam_inference_with_cropped(predictor, cropped_path, data["points"][0], mask_name, output_dir, cropped_output_dir)
                if new_cropped:
                    cropped_images[mask_name] = new_cropped


if __name__ == "__main__":
    # 配置路径
    points_json = "/home/zqin74/RGB/point_prompts8.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"

    image_root_dir = "/home/zqin74/RGB/Rasp2"
    output_dir = "/home/zqin74/RGB/Seg_Rap2"
    cropped_output_dir = "/home/zqin74/RGB/Cropped_Images"

    # 读取关键点数据
    with open(points_json, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)

    # 运行 SAM 处理图像序列
    process_images_in_sequence(
        prompts_dict=prompts_dict,
        sam_checkpoint=sam_checkpoint_path,
        model_type=sam_model_type,
        image_root=image_root_dir,
        output_dir=output_dir,
        cropped_output_dir=cropped_output_dir
    )
