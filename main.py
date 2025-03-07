from src.sam_segmentation import run_sam
from src.bbox_extraction import generate_bboxes
from src.bad_case_analysis import filter_bad_cases
from src.bbox_iou_analysis import process_missing_bboxes
from src.sam_roi_segmentation import run_sam_on_rois
from src.mask_projection import process_mask_folder


import json

def main():
    # File paths
    points_json = "/home/zqin74/RGB/point_prompts8.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    image_root_dir = "/home/zqin74/RGB/Rasp2"
    output_dir = "/home/zqin74/RGB/Seg_Rap2"
    mask_json_output = "/home/zqin74/RGB/ts/masks2.json"
    bbox_json_output = "/home/zqin74/RGB/ts/bboxes2.json"
    bad_case_json_output = "/home/zqin74/RGB/ts/bad_case2.json"
    missing_bbox_json_output = "/home/zqin74/RGB/ts/missing_bboxes2.json"
    crop_output_dir = "/home/zqin74/RGB/IOU/crop2"
    bad_case_folder = "/home/zqin74/RGB/ts/missing_bbox"
    sam_model_type = "vit_h"
    
    # Load JSON data
    with open(points_json, 'r', encoding="utf-8") as f:
        prompts_list = json.load(f) 

    # Step 1: Run SAM segmentation to generate masks
    # run_sam(prompts_list, sam_checkpoint_path, image_root=image_root_dir, output_dir=output_dir, mask_json_output=mask_json_output)

    # Step 2: Generate bounding boxes from masks
    # generate_bboxes(mask_json_output, bbox_json_output)

    # Step 3: Filter out abnormal bounding boxes based on aspect ratio
    # filter_bad_cases(bbox_json_output, bad_case_json_output)

    # Step 4: Process missing bounding boxes
    # process_missing_bboxes(
    #     bad_case_json_output, bbox_json_output, image_root_dir,
    #     bad_case_folder, crop_output_dir, missing_bbox_json_output,
    #     iou_threshold=0.8, padding=15)

     # Step 5: 在缺失的 bounding boxes 位置运行 SAM 分割
    # run_sam_on_rois(crop_output_dir, bbox_json_output, sam_checkpoint_path, sam_model_type)

    # Step 6: 将 ROI masks 重新映射回原始图像尺寸，并存放到 output_dir
    mask_folder = "/home/zqin74/RGB/IOU"  # 指定 mask 文件夹路径
    original_image_shape = (1080, 1920)  # 设定原始图像尺寸

    process_mask_folder(crop_output_dir, original_image_shape, output_dir)


if __name__ == "__main__":
    main()
