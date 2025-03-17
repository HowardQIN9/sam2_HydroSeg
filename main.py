from src.sam_segmentation import run_sam
from src.bbox_extraction import generate_bboxes
from src.bad_case_analysis import filter_bad_cases
from src.bbox_iou_analysis import process_missing_bboxes
# from src.sam_roi_segmentation import process_masks_with_padding# run_sam_on_rois
from src.mask_padding import process_masks_with_padding
from src.mask_projection import process_mask_folder
import json

def main(points_json, sam_checkpoint_path, image_root_dir, output_dir, 
         mask_json_output, bbox_json_output, bad_case_json_output, 
         missing_bbox_json_output, crop_output_dir, sam_model_type):
    # 你的 main 函数逻辑
    print("Running with:")
    print("  points_json =", points_json)
    print("  sam_checkpoint_path =", sam_checkpoint_path)
    print("  image_root_dir =", image_root_dir)
    print("  output_dir =", output_dir)
    print("  mask_json_output =", mask_json_output)
    print("  bbox_json_output =", bbox_json_output)
    print("  bad_case_json_output =", bad_case_json_output)
    print("  missing_bbox_json_output =", missing_bbox_json_output)
    print("  crop_output_dir =", crop_output_dir)
    print("  sam_model_type =", sam_model_type)
    print("-----------------------------------\n")
   # Load JSON data
    with open(points_json, 'r', encoding="utf-8") as f:
        prompts_list = json.load(f) 

    # # Step 1: Run SAM segmentation to generate masks
    # run_sam(prompts_list, sam_checkpoint_path, image_root=image_root_dir, output_dir=output_dir, mask_json_output=mask_json_output)

    # # Step 2: Generate bounding boxes from masks
    generate_bboxes(mask_json_output, bbox_json_output)

    # # Step 3: Filter out abnormal bounding boxes based on aspect ratio
    start_date = "2024-10-23"
    filter_bad_cases(bbox_json_output, bad_case_json_output,start_date)

    # Step 4: Process missing bounding boxes
    process_missing_bboxes(
         bad_case_json_output, bbox_json_output, image_root_dir,
         crop_output_dir, missing_bbox_json_output,
         iou_threshold=0.8, padding=15)

     # Step 5: 在缺失的 bounding boxes 位置运行 SAM 分割
    # run_sam_on_rois(crop_output_dir, bbox_json_output, sam_checkpoint_path, sam_model_type)

    # original_size = (800, 600)  # 替换为原始图像的尺寸 (width, height)

    # 设定 mask padding 处理的 kernel 大小
    kernel_size = 5  # 可以根据目标大小的变化情况调整
    # process_masks_with_padding(crop_output_dir, mask_json_output, original_size, kernel_size)
    max_lookback_days = 4
    process_masks_with_padding(missing_bbox_json_output, mask_json_output ,output_dir, kernel_size, max_lookback_days)



    # Step 6: 将 ROI masks 重新映射回原始图像尺寸，并存放到 output_dir
    #nmask_folder = "/home/zqin74/MSI/IOU"  # 指定 mask 文件夹路径

    # process_mask_folder(crop_output_dir, original_size, output_dir)


if __name__ == "__main__":
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"

    # i 从 1 跑到 5，并且 points_json 的数字从 7 到 11
    for i in range(1, 6):
        if i>1:
            break

        points_json = f"/home/zqin74/MSI/point_prompt/point_prompts{i}.json"
        image_root_dir = f"/home/zqin74/MSI/MS{i}"
        output_dir = f"/home/zqin74/MSI/Seg_Rap{i}"
        mask_json_output = f"/home/zqin74/MSI/ts/masks{i}.json"
        bbox_json_output = f"/home/zqin74/MSI/ts/bboxes{i}.json"
        bad_case_json_output = f"/home/zqin74/MSI/ts/bad_case{i}.json"
        missing_bbox_json_output = f"/home/zqin74/MSI/ts/missing_bboxes{i}.json"
        crop_output_dir = f"/home/zqin74/MSI/IOU/crop{i}"

        # 调用 main 函数
        main(points_json,
             sam_checkpoint_path,
             image_root_dir,
             output_dir,
             mask_json_output,
             bbox_json_output,
             bad_case_json_output,
             missing_bbox_json_output,
             crop_output_dir,
             sam_model_type)



# import os
# import json
# import datetime
# from src.fix_bad import process_bad_cases  # 确保此模块包含完整代码
# from src.mask_projection import process_mask_folder  # 负责将 mask 重新映射到原图
# from src.run_sam import run_sam  # 初次运行 SAM 进行分割

# def main():
#     # **路径设置**
#     points_json = "/home/zqin74/RGB/point_prompts9.json"
#     sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
#     image_root_dir = "/home/zqin74/RGB/Rasp3"
#     output_dir = "/home/zqin74/RGB/v2/Seg_Rap3"
#     mask_json_output = "/home/zqin74/RGB/v2/masks3.json"  # **存储最终所有 mask**
#     bad_case_json_output = "/home/zqin74/RGB/v2/bad_case3.json"
#     crop_output_dir = "/home/zqin74/RGB/v2/crop3"
#     projected_mask_output_dir = "/home/zqin74/RGB/v2/projected_masks"
#     sam_model_type = "vit_h"

#     # **设定 bad case 处理的起始日期**
#     start_date = datetime.datetime(2024, 10, 21)  # 例如 2024 年 3 月 27 日之后进行 bbox 过滤

#     # **初次运行 SAM 生成 mask**
#     if not os.path.exists(mask_json_output):
#         print("[INFO] Running SAM for the first time...")
#         with open(points_json, "r", encoding="utf-8") as f:
#             prompts_list = json.load(f)

#         run_sam(
#             prompts_list=prompts_list,
#             sam_checkpoint=sam_checkpoint_path,
#             model_type=sam_model_type,
#             image_root=image_root_dir,
#             output_dir=output_dir,
#             mask_json_output=mask_json_output,
#             bad_case_json_output=bad_case_json_output,
#             start_date=start_date
#         )

#     # **迭代修正 bad case**
#     print("[INFO] Starting bad case correction loop...")
#     process_bad_cases(
#         bad_case_json=bad_case_json_output,
#         masks_json_path=mask_json_output,
#         image_folder=image_root_dir,
#         save_folder=output_dir,
#         crop_output_dir=crop_output_dir,
#         sam_checkpoint=sam_checkpoint_path,
#         model_type=sam_model_type
#     )

#     # **映射最终 mask 到原始尺寸**
#     print("[INFO] Projecting corrected masks back to original size...")
#     original_image_shape = (1080, 1920)  # **确保与原始图像尺寸匹配**
#     process_mask_folder(mask_folder=crop_output_dir, original_image_shape=original_image_shape, output_folder=projected_mask_output_dir)

#     print("[SUCCESS] All masks are processed and saved!")

# if __name__ == "__main__":
#     main()
