# ==============================================
# 钢材表面缺陷检测可视化工具 (Streamlit版)
# 功能：上传图片 → 一键检测 → 可视化展示结果
# ==============================================

# 🔴 第一步：彻底关闭所有联网，解决SSL报错（必须放最顶部）
import os
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["ULTRALYTICS_SKIP_VERSION_CHECK"] = "True"
os.environ["ULTRALYTICS_HUB"] = ""

import cv2
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------- 模型配置（你的路径，不用改）--------------------------
MODEL_PATH = "best.pt"
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# --------------------------------------------------------------------------------

# 页面样式
st.set_page_config(page_title="钢材缺陷检测", layout="wide")
st.title("🛠️ 钢材表面缺陷智能检测系统")
st.caption("基于 YOLOv8 | 支持6类缺陷检测：裂纹/夹杂物/斑块/点蚀/氧化皮/划痕")

# 侧边栏参数调节
st.sidebar.header("⚙️ 检测参数设置")
conf_threshold = st.sidebar.slider("置信度阈值", 0.05, 0.5, 0.1, 0.01)
iou_threshold = st.sidebar.slider("IOU阈值", 0.1, 0.8, 0.2, 0.01)
imgsz = st.sidebar.selectbox("输入尺寸", [640, 1280], index=1)

# 缓存加载模型（只加载一次）
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH, task="detect")

model = load_model()

# 主功能：图片上传
uploaded_file = st.file_uploader("📸 上传钢材图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 读取图片
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    
    # 显示原图
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图片")
        st.image(img, use_column_width=True)
    
    # 检测按钮
    if st.button("🚀 开始缺陷检测"):
        with st.spinner("模型推理中..."):
            # 模型推理
            results = model.predict(
                source=img_np,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                multi_scale=True,
                max_det=100,
                save=False,
                verbose=False
            )

            # 画框（解决乱码）
            result_img = img_np.copy()
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id]

                # 画框 + 文字
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(result_img, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示结果
            with col2:
                st.subheader("检测结果")
                st.image(result_img, use_column_width=True)
            
            # 统计结果
            defect_count = len(results[0].boxes)
            st.success(f"✅ 检测完成！共发现 **{defect_count}** 个缺陷")

            # 保存结果
            cv2.imwrite(f"detect_{uploaded_file.name}", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            st.caption(f"结果已自动保存：detect_{uploaded_file.name}")

st.markdown("---")
st.markdown("💡 使用说明：上传图片 → 调节参数 → 点击检测 → 实时查看缺陷位置")
# ==============================================
# 钢材表面缺陷检测可视化工具 (Streamlit版)
# 功能：上传图片 → 一键检测 → 可视化展示结果
# ==============================================


# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO

# # 页面配置
# st.set_page_config(page_title="钢材缺陷检测", layout="wide")
# st.title("🛠️ 钢材表面缺陷检测系统 (YOLOv8)")

# # 加载模型
# @st.cache_resource
# def load_model():
#     model = YOLO("best.pt")
#     return model

# model = load_model()

# # 上传图片
# uploaded_file = st.file_uploader("上传钢材图片", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # 读取图片（纯 Pillow，无 cv2）
#     image = Image.open(uploaded_file).convert("RGB")
    
#     # 显示原图
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("原图")
#         st.image(image, use_column_width=True)
    
#     # 模型推理
#     with st.spinner("检测中..."):
#         results = model(image)
    
#     # 获取检测结果图片
#     result_img = results[0].plot()
    
#     # 显示结果
#     with col2:
#         st.subheader("检测结果")
#         st.image(result_img, channels="RGB", use_column_width=True)
    
#     # 输出检测信息
#     st.success("检测完成！")