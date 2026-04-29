from ultralytics import YOLO

# 加载预训练模型
model = YOLO("../yolov8n.pt")

# 开始训练（已经适配CPU+你的路径，直接用！）
results = model.train(
    data="neu_defect.yaml",  # 直接写文件名，因为和train.py在同一个目录
    epochs=50,
    imgsz=640,
    batch=1,
    device="cpu",
    workers=0,  # CPU训练必须设为0！
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    hsv_h=0.015,
    hsv_s=0.2,
    hsv_v=0.2,
    degrees=5,
    translate=0.05,
    scale=0.1,
    mosaic=0.0,
    mixup=0.0,
    flipud=0.2,
    fliplr=0.5,
    project="../runs/train",  # 保存到根目录的runs里
    name="steel_defect_detector",
    exist_ok=True
)

print("✅ 训练完成！模型保存在 runs/train/steel_defect_detector/weights/best.pt")