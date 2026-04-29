import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# ===================== 路径自动匹配，不用改 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_img_dir = os.path.join(BASE_DIR, "dataset", "NEU-DET", "IMAGES")
raw_anno_dir = os.path.join(BASE_DIR, "dataset", "NEU-DET", "ANNOTATIONS")
save_dir = os.path.join(BASE_DIR, "dataset", "neu_defect")

# 缺陷类别映射，和官方标注完全匹配
class_map = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}
# =================================================================

# 1. 创建YOLO标准文件夹
for split in ["train", "val"]:
    os.makedirs(os.path.join(save_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, split, "labels"), exist_ok=True)

# 2. 按8:2拆分训练集/验证集
all_imgs = [f for f in os.listdir(raw_img_dir) if f.endswith(".jpg")]
train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42)
print(f"训练集：{len(train_imgs)} 张")
print(f"验证集：{len(val_imgs)} 张")

# 3. XML官方标注 转 YOLO格式标注
def xml_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        cls_id = class_map[cls]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 转成YOLO归一化格式
        xc = (xmin + xmax) / 2 / img_w
        yc = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return labels

# 4. 批量处理数据集
def process_set(img_list, split):
    for img in img_list:
        # 复制图片到对应文件夹
        img_src = os.path.join(raw_img_dir, img)
        img_dst = os.path.join(save_dir, split, "images", img)
        shutil.copy(img_src, img_dst)

        # 转换标注文件
        xml_file = img.replace(".jpg", ".xml")
        xml_path = os.path.join(raw_anno_dir, xml_file)
        labels = xml_to_yolo(xml_path, 200, 200) # NEU数据集图片固定200×200

        # 保存YOLO格式标注
        label_dst = os.path.join(save_dir, split, "labels", img.replace(".jpg", ".txt"))
        with open(label_dst, "w") as f:
            f.write("\n".join(labels))

# 执行处理
process_set(train_imgs, "train")
process_set(val_imgs, "val")

# 运行完成提示
print("="*50)
print("✅ 数据集处理完成！")
print(f"✅ 输出路径：{save_dir}")
print("✅ 可以直接开始训练模型！")