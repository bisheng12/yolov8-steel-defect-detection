# # 🔴 先关闭所有联网行为
# import os
# os.environ["ULTRALYTICS_OFFLINE"] = "1"
# os.environ["ULTRALYTICS_SKIP_VERSION_CHECK"] = "True"

# import cv2
# from ultralytics import YOLO

# # -------------------------- 路径已写死，不用改！ --------------------------
# MODEL_PATH = "./runs/runs/train/steel_defect_detector/weights/best.pt"
# IMG_FOLDER = "D:/defect_detection_project/dataset/neu_defect/val/images/"
# SAVE_FOLDER = "D:/defect_detection_project/code/batch_results_optimized/"
# # 关键参数调整：降低置信度，保留更多缺陷
# CONF_THRESHOLD = 0.1  # 降到0.1，先把所有可能的缺陷都保留下来
# IOU_THRESHOLD = 0.5   # 提高IOU阈值，减少重叠框被误删
# CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# # ---------------------------------------------------------------------------

# # 创建保存文件夹
# os.makedirs(SAVE_FOLDER, exist_ok=True)

# # 加载模型
# model = YOLO(MODEL_PATH, task="detect")

# # 遍历所有图片
# img_list = os.listdir(IMG_FOLDER)
# print(f"📸 共找到 {len(img_list)} 张图片，开始批量检测...")

# for i, img_name in enumerate(img_list):
#     img_path = os.path.join(IMG_FOLDER, img_name)
#     print(f"正在检测第 {i+1}/{len(img_list)} 张：{img_name}")
    
#     # 推理：开启verbose看每个缺陷的置信度
#     results = model.predict(
#         source=img_path,
#         conf=CONF_THRESHOLD,
#         iou=IOU_THRESHOLD,
#         agnostic_nms=True,  # 不同类别的框不会互相过滤，避免漏检
#         save=False,
#         verbose=True
#     )
    
#     # 用OpenCV画框，避免乱码
#     img = cv2.imread(img_path)
#     for result in results:
#         for box in result.boxes:
#             # 获取框坐标和置信度
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             class_id = int(box.cls[0])
#             class_name = CLASS_NAMES[class_id]
            
#             # 画绿色框
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             # 画文字（白色+黑色背景，清晰不重叠）
#             label = f"{class_name} {conf:.2f}"
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(img, (x1, y1-20), (x1+w, y1), (0, 0, 0), -1)
#             cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
#     # 保存结果
#     save_path = os.path.join(SAVE_FOLDER, f"result_{img_name}")
#     cv2.imwrite(save_path, img)

# print(f"✅ 批量检测完成！所有结果已保存到：{SAVE_FOLDER}")
# 🔴 先关闭所有联网行为，避免SSL报错
import os
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["ULTRALYTICS_SKIP_VERSION_CHECK"] = "True"

import cv2
from ultralytics import YOLO

# -------------------------- 路径已写死，不用改！ --------------------------
MODEL_PATH = "./runs/runs/train/steel_defect_detector/weights/best.pt"
IMG_FOLDER = "D:/defect_detection_project/dataset/neu_defect/val/images/"
SAVE_FOLDER = "D:/defect_detection_project/code/batch_results_fixed/"
# 【核心参数优化，专门解决你的问题】
CONF_THRESHOLD = 0.08  # 降到0.08，把极低置信度的小缺陷也捞出来
IOU_THRESHOLD = 0.2    # 大幅降低IOU阈值，禁止把多个小框合并成大框
MAX_DET = 100           # 单图最多检测100个框，适配密集的点蚀缺陷
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# ---------------------------------------------------------------------------

# 创建保存文件夹
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 加载模型
model = YOLO(MODEL_PATH, task="detect")

# 遍历所有图片
img_list = os.listdir(IMG_FOLDER)
print(f"📸 共找到 {len(img_list)} 张图片，开始优化版批量检测...")

for i, img_name in enumerate(img_list):
    img_path = os.path.join(IMG_FOLDER, img_name)
    print(f"正在检测第 {i+1}/{len(img_list)} 张：{img_name}")
    
    # 【核心：推理参数全优化，专门适配钢材微小缺陷】
    results = model.predict(
        source=img_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DET,
        agnostic_nms=False,  # 关闭类别无关NMS，避免不同缺陷的框乱合并
        multi_scale=True,    # 开启多尺度检测，大幅提升小缺陷检出率
        imgsz=1280,          # 输入尺寸放大到1280，小缺陷看得更清楚
        save=False,
        verbose=True
    )
    
    # 画框+标注，避免乱码
    img = cv2.imread(img_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            
            # 画红色细框，避免大框遮挡缺陷
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # 标注文字，小字号适配密集缺陷
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # 保存结果
    save_path = os.path.join(SAVE_FOLDER, f"fixed_{img_name}")
    cv2.imwrite(save_path, img)

print(f"✅ 优化版检测完成！所有结果已保存到：{SAVE_FOLDER}")