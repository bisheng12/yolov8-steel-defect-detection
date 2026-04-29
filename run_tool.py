from defect_detector import SteelDefectDetector

if __name__ == "__main__":
    # 1. 初始化工具（自动加载模型、关闭联网）
    detector = SteelDefectDetector(config_path="config.yaml")
    
    # -------------------------- 选择检测模式（二选一，注释掉另一个） --------------------------
    
    # 模式1：单张图片检测
    # detector.predict_single(
    #     img_path="D:/defect_detection_project/dataset/neu_defect/val/images/crazing_1.jpg"
    # )
    
    # 模式2：批量图片检测（默认开启）
    detector.predict_batch(
        img_folder="D:/defect_detection_project/dataset/neu_defect/val/images/"
    )