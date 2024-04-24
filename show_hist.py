import os, tqdm
import matplotlib.pyplot as plt
import numpy as np

def read_yolo_labels(label_path):
    # 读取标签文件
    with open(label_path, "r") as f:
        labels = f.readlines()

    # 解析标签
    boxes = []
    for label in labels:
        # 将标签分割成各个部分
        parts = label.split()

        # 获取目标类别
        class_id = int(parts[0])

        # 获取目标中心点坐标
        x_center = float(parts[1])
        y_center = float(parts[2])

        # 获取目标宽高
        width = float(parts[3])
        height = float(parts[4])

        # 计算目标的左上角和右下角坐标
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # 将目标信息添加到列表中
        boxes.append([x_min, y_min, x_max, y_max, class_id])

    return boxes

def show_width_height_data(boxes):
    # 计算目标的宽度和高度
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]

    # 统计每个宽度和高度区间的目标数量
    width_bins = np.arange(0, 0.05 + 0.005, 0.005)
    height_bins = np.arange(0, 0.05 + 0.005, 0.005)
    width_counts, _ = np.histogram(widths, bins=width_bins)
    height_counts, _ = np.histogram(heights, bins=height_bins)
    
    # 输出统计结果
    print(f"宽度统计: mean:{np.mean(widths):.4f}")
    print("-" * 20)
    print("区间 | 数量")
    print("---|---|")
    for i in range(len(width_bins) - 1):
        print(f"{width_bins[i]:.4f} - {width_bins[i + 1]:.4f} | {width_counts[i]}")

    print(f"\n高度统计: mean:{np.mean(heights):.4f}")
    print("-" * 20)
    print("区间 | 数量")
    print("---|---|")
    for i in range(len(height_bins) - 1):
        print(f"{height_bins[i]:.4f} - {height_bins[i + 1]:.4f} | {height_counts[i]}")

if __name__ == "__main__":
    # 标签文件夹路径
    label_dir = "/root/data_ssd/WIDER-FACE/train"

    # 统计所有目标的宽度和高度
    all_boxes = []
    for filename in tqdm.tqdm(os.listdir(label_dir)):
        if 'txt' in filename:
            label_path = os.path.join(label_dir, filename)
            boxes = read_yolo_labels(label_path)
            all_boxes.extend(boxes)

    # 绘制柱状图
    show_width_height_data(all_boxes)