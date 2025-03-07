import os
import re
import cv2
import json
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import defaultdict
from segment_anything import SamPredictor, sam_model_registry

def get_previous_date(date_str):
    """
    计算前一天的日期，自动处理跨月、跨年
    :param date_str: 日期字符串，例如 '20241029'
    :return: 前一天的日期字符串，例如 '20241028'
    """
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    prev_date_obj = date_obj - timedelta(days=1)
    return prev_date_obj.strftime("%Y%m%d")

def get_aspect_ratio(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return h / w

def get_bounding_box(mask, padding=10):
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        return None
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, mask.shape[1])
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, mask.shape[0])

    return (x_min, y_min, x_max, y_max)

def crop_image(image, mask, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped_img = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    return cropped_img, cropped_mask

def run_sam_with_prompts(
    prompts_dict,
    sam_checkpoint,
    model_type="vit_h",
    image_root="",
    output_dir="masks",
    min_ratio=5/7,
    max_ratio=7/5,
    start_date=None
):
    """
    :param prompts_dict: 从 JSON 读入的点提示信息
    :param sam_checkpoint: SAM 模型权重
    :param model_type: SAM 模型类型 (vit_h / vit_l / vit_b 等)
    :param image_root: 原始图像所在文件夹
    :param output_dir: 掩码输出目录
    :param min_ratio: 长宽比下限
    :param max_ratio: 长宽比上限
    :param start_date: 从哪一天（含）开始执行“前一天掩码辅助”检查，如 "20241101"
    """
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 如果传了 start_date，则转成 datetime 以便比较
    start_date_obj = None
    if start_date is not None:
        start_date_obj = datetime.strptime(start_date, "%Y%m%d")

    os.makedirs(output_dir, exist_ok=True)

    for img_basename, image_data in prompts_dict.items():
        image_path = os.path.join(image_root, img_basename)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Failed to read image: {image_path}, skipping...")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 每处理一张图先把 predictor 设置到当前图像
        predictor.set_image(img_rgb)

        # 获取当前图像的日期（比如 image_20241112_130006_T11F8.jpg -> 20241112）
        match = re.search(r"image_(\d{8})_", img_basename)
        current_date_str = None
        if match:
            current_date_str = match.group(1)

        for i, (point, mask_name) in enumerate(zip(image_data["points"], image_data["mask_names"])):
            try:
                point_coords = np.array([point], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)

                # 构造当前 mask 的输出路径
                base_name = os.path.splitext(img_basename)[0]
                mask_filename = f"{base_name}_{mask_name}_mask.png"
                mask_path = os.path.join(output_dir, mask_filename)

                # 1) 如果当前 mask 文件不存在，就用 point prompt 生成一个初始 mask
                mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_array is None:
                    # 用 SAM 根据点提示生成初步分割
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True  # 你也可以改成 False
                    )
                    # 选分数最高的那个
                    best_idx = np.argmax(scores)
                    mask_array = (masks[best_idx] * 255).astype(np.uint8)

                    cv2.imwrite(mask_path, mask_array)
                    print(f"Generated initial mask: {mask_path}")

                # 2) 检查长宽比
                aspect_ratio = None
                # 我们已经有 mask_array 了，可以不再从磁盘读
                # 直接根据 mask_array.shape 来计算
                h, w = mask_array.shape[:2]
                if w != 0:
                    aspect_ratio = h / w

                # 3) 如果长宽比超出范围，需要看当前日期是否 >= start_date
                #    并尝试去使用前一天的 mask 做修正
                if aspect_ratio and (aspect_ratio < min_ratio or aspect_ratio > max_ratio):
                    print(f"Aspect ratio {aspect_ratio:.2f} out of range for {mask_name}, refining mask...")

                    # 若没有解析到日期，或者当前日期 < start_date，就跳过“前一天”修正逻辑
                    if not current_date_str:
                        print(f"Cannot extract date from {img_basename}, skipping correction...")
                    elif start_date_obj and datetime.strptime(current_date_str, "%Y%m%d") < start_date_obj:
                        print(f"Current date {current_date_str} is before start_date {start_date}, skipping correction...")
                    else:
                        # 执行前一天掩码辅助修正
                        previous_date_str = get_previous_date(current_date_str)
                        # 在 output_dir 里查找前一天同一个 mask_name 的文件
                        prev_mask_filename = f"image_{previous_date_str}_*_mask.png"
                        prev_mask_path = None
                        for file in os.listdir(output_dir):
                            # 用 mask_name 匹配
                            # 例如 mask_name = T11F8 -> image_20241111_*_T11F8_mask.png
                            # 但是你的文件命名是 f"image_{date}_130006_T11F8_mask.png"，
                            # 所以可以用 mask_name 来替换 * 试试:
                            if re.match(prev_mask_filename.replace("*", mask_name), file):
                                prev_mask_path = os.path.join(output_dir, file)
                                break

                        if prev_mask_path is None or not os.path.exists(prev_mask_path):
                            print(f"Previous mask for {mask_name} on {previous_date_str} not found, skipping correction...")
                        else:
                            prev_mask = cv2.imread(prev_mask_path, cv2.IMREAD_GRAYSCALE)
                            if prev_mask is None:
                                print(f"Failed to read previous mask {prev_mask_path}, skipping correction...")
                            else:
                                # 获取 BBox 并裁剪当前帧
                                bbox = get_bounding_box(prev_mask, padding=10)
                                if bbox is None:
                                    print(f"Invalid bounding box for {mask_name}, skipping correction...")
                                else:
                                    # 注意，要让SAM预测时再 set_image，这次是裁剪后的图
                                    # 不过由于想要贴回原图，需要先记住 bbox
                                    # 先裁剪
                                    cropped_img, _ = crop_image(img_rgb, mask_array, bbox)

                                    # 重新在裁剪图上做分割
                                    predictor.set_image(cropped_img)
                                    # 这里演示使用 bounding box 的用法
                                    # 你也可以同时加 point prompt
                                    # predictor.predict(box=bbox) 会把 bbox 当 prompt
                                    # 但是 docs 中 box 要用 (xmin, ymin, xmax, ymax) in the input image space
                                    # 如果只是在crop上，可直接 predict 不一定要 bbox
                                    # 这里做演示:
                                    # --------------------------------------------------
                                    # 下面这个 predict 的 box 参数指的是 *裁剪后图像* 的坐标系 (0,0 到 w_c,h_c)
                                    # 但 SAM 公式是：box 在原图坐标系中 -> 先 set_image(img_rgb) 再 box=bbox
                                    # 这里可能得改回到把 set_image(img_rgb) 再 predict(box=bbox)
                                    # 否则坐标会对不上
                                    # 为了简单，直接用 point prompt 来 refine 也行
                                    predictor.set_image(cropped_img)
                                    masks_new, scores_new, _ = predictor.predict(
                                        point_coords=np.array([[w//2, h//2]], dtype=np.float32),
                                        point_labels=np.array([1], dtype=np.int32),
                                        multimask_output=True
                                    )
                                    best_idx_new = np.argmax(scores_new)
                                    new_cropped_mask = (masks_new[best_idx_new] * 255).astype("uint8")

                                    # 还原到原图尺寸
                                    new_mask = np.zeros_like(mask_array)
                                    x_min, y_min, x_max, y_max = bbox
                                    # 如果 new_cropped_mask 与 bbox 尺寸不一致，需要对齐
                                    # 如果上面 predictor.predict 时只用了 center point prompt，
                                    # 并且图片尺寸没有变，就可以直接贴
                                    h_c, w_c = new_cropped_mask.shape[:2]
                                    new_mask[y_min:y_min+h_c, x_min:x_min+w_c] = new_cropped_mask

                                    mask_array = new_mask  # 更新当前 mask

                                    # 再把 predictor 恢复原图
                                    predictor.set_image(img_rgb)

                # 4) 过滤噪声：保留最大连通域
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    filtered_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                else:
                    filtered_mask = mask_array

                # 5) 保存最终 mask
                save_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(save_path, filtered_mask)
                print(f"Saved mask: {save_path}")

            except Exception as e:
                print(f"Error processing {img_basename} - {mask_name}: {str(e)}")

if __name__ == "__main__":
    points_json = "/home/zqin74/RGB/point_prompts7.json"
    sam_checkpoint_path = "/home/zqin74/RGB/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    image_root_dir = "/home/zqin74/RGB/Rasp1"
    output_dir = "/home/zqin74/RGB/Seg_Rap1"

    # 例如，只有当图片日期 >= 20241110 才去做“前一天掩码”对齐与检查
    start_date = "20241023"

    with open(points_json, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)

    run_sam_with_prompts(
        prompts_dict=prompts_dict,
        sam_checkpoint=sam_checkpoint_path,
        model_type=sam_model_type,
        image_root=image_root_dir,
        output_dir=output_dir,
        start_date=start_date  # 新增参数
    )
