import os
import cv2
import yaml
from ultralytics import YOLO

class SteelDefectDetector:
    def __init__(self, config_path="config.yaml"):
        # 🔴 必须放在最前面：彻底关闭所有联网行为，避免SSL报错
        self._disable_network()
        
        # 加载配置文件
        self.cfg = self._load_config(config_path)
        
        # 加载模型
        print(f"正在加载模型：{self.cfg['model_path']}")
        self.model = YOLO(self.cfg['model_path'], task="detect")
        print("✅ 模型加载成功！")

    def _disable_network(self):
        """彻底关闭ultralytics的所有联网行为"""
        os.environ["ULTRALYTICS_OFFLINE"] = "1"
        os.environ["ULTRALYTICS_SKIP_VERSION_CHECK"] = "True"
        os.environ["ULTRALYTICS_HUB"] = ""
        os.environ["ULTRALYTICS_CONFIG_DIR"] = "./.ultralytics"
        
        # 禁用requests的SSL验证，双重保险
        import requests
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        requests.Session().verify = False

    def _load_config(self, config_path):
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _draw_boxes(self, img, boxes):
        """用OpenCV画框，避免乱码"""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.cfg['class_names'][class_id]
            
            # 画红色细框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # 画文字（白色+黑色背景）
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1-15), (x1+w, y1), (0, 0, 0), -1)
            cv2.putText(img, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img

    def predict_single(self, img_path, save_path=None):
        """单张图片检测"""
        print(f"正在检测单张图片：{img_path}")
        img = cv2.imread(img_path)
        
        # 推理
        results = self.model.predict(
            source=img_path,
            conf=self.cfg['conf_threshold'],
            iou=self.cfg['iou_threshold'],
            max_det=self.cfg['max_det'],
            imgsz=self.cfg['imgsz'],
            multi_scale=self.cfg['multi_scale'],
            save=False,
            verbose=False
        )
        
        # 画框
        for result in results:
            img = self._draw_boxes(img, result.boxes)
        
        # 保存
        if save_path is None:
            save_path = f"result_{os.path.basename(img_path)}"
        cv2.imwrite(save_path, img)
        print(f"✅ 单张检测完成！结果已保存到：{save_path}")
        return img

    def predict_batch(self, img_folder, save_folder=None):
        """批量图片检测"""
        if save_folder is None:
            save_folder = self.cfg['save_folder']
        os.makedirs(save_folder, exist_ok=True)
        
        img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"📸 共找到 {len(img_list)} 张图片，开始批量检测...")
        
        for i, img_name in enumerate(img_list):
            img_path = os.path.join(img_folder, img_name)
            save_path = os.path.join(save_folder, f"result_{img_name}")
            
            print(f"[{i+1}/{len(img_list)}] 正在检测：{img_name}")
            
            # 推理
            results = self.model.predict(
                source=img_path,
                conf=self.cfg['conf_threshold'],
                iou=self.cfg['iou_threshold'],
                max_det=self.cfg['max_det'],
                imgsz=self.cfg['imgsz'],
                multi_scale=self.cfg['multi_scale'],
                save=False,
                verbose=False
            )
            
            # 画框+保存
            img = cv2.imread(img_path)
            for result in results:
                img = self._draw_boxes(img, result.boxes)
            cv2.imwrite(save_path, img)
        
        print(f"✅ 批量检测完成！所有结果已保存到：{save_folder}")