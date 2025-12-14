import streamlit as st
import pandas as pd

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision


st.set_page_config(page_title="SmartVision AI - Intelligent Multi-Class Object Recognition System", layout="wide")
st.sidebar.title("📘 SmartVision AI")

st.sidebar.markdown("---")

st.title("🤖 SmartVision AI - Intelligent Multi-Class Object Recognition System")
st.markdown("---")

page = st.sidebar.radio("Go to", ["🏠 Home", "🖼️ Image Classification", "📦 Object Detection", "📊 Model Performance", "📸 Live Webcam Detection","ℹ️ About"])



#------------------------------------------------Home Page----------------------------------------------------------------------------------------

if page == "🏠 Home":
    st.subheader("📌 Project Overview")
    st.markdown("""
        **SmartVision AI** is an intelligent computer vision system that performs real-time object detection 
        using a custom-trained **YOLO model**.  
        The system allows users to upload images and automatically identifies objects by drawing bounding boxes,
        class labels, and confidence scores.

        The goal of this project is to demonstrate an **end-to-end AI pipeline** — from model training 
        to optimized inference and visualization.
        """)
    
    st.info("✨ This project is designed to showcase practical skills in Deep Learning, Computer Vision, and Model Deployment, with a focus on performance optimization and clean output presentation.")
    st.markdown("---")
    st.subheader("🚀 Key Features")
    st.markdown("""
            ➤ 🔍 **Accurate Object Detection** using a trained YOLO model  
            ➤ 📦 **Bounding Boxes & Labels** on detected objects  
            ➤ 📊 **Confidence Scores** for every prediction  
            ➤ 🧠 **Optional CNN-based verification**  
            ➤ ⚡ **Optimized CNNs** (VGG16, ResNet50, MobileNetV2, EfficientNet-B0)
            """)

    
    st.markdown("---")
    st.subheader("📝 Instructions for Users")
    st.text("""
                    ➤  🔍 Navigate to the Detection page
                    ➤  📦 Upload an image (JPG / PNG format)
                    ➤  📊 Wait for the model to process the image
                    ➤  🧠 View the output image with bounding boxes and labels
                    ➤  ⚡ Check confidence scores for each detected object
                """)
    st.info("⚠️ For best results, use clear images with good lighting and visible objects.")
    
    st.markdown("---")
    st.subheader("🖼️ Sample Demo Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image("img.png", caption="YOLO Detection Example 1")
    with col2:
        st.image("img1.png", caption="YOLO Detection Example 2")

        
    
#----------------------------------------------------------------------------------------------------------------------------------
# we have already trained these models in the collab and using the state.dict(),after saving .here i am using the path of models

Classes= ['airplane', 'banana', 'bear', 'bicycle', 'bird', 'bowl', 'bus', 'cake', 'car', 'cat', 'dog', 'elephant', 'horse', 'laptop', 'motorcycle', 'mouse', 'parking meter', 'person', 'potted plant', 'sheep', 'toilet', 'traffic light', 'truck', 'tv', 'wine glass']
NUM_CLASSES = len(Classes)  # 25

import torch
import torch.nn as nn
import torchvision.models as models


# vgg16
@st.cache_resource
def load_custom_vgg16():
    model = models.vgg16(pretrained=False)

    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),

        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),

        nn.Linear(512, 25)   # number of classes
    )

    model.load_state_dict(
        torch.load(
            "models/vgg16_smartvision.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model

# RestNet50
@st.cache_resource
def load_custom_restnet50():
    model=models.resnet50(pretrained=False)
    
    #  CUSTOM CLASSIFICATION HEAD
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    
    model.load_state_dict(
        torch.load(
            "models/smartvision_resnet50.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model

# Mobilenet_v2
@st.cache_resource
def load_custom_mobilenetv2():
    model=models.mobilenet_v2(pretrained=False)
    
    #  CUSTOM CLASSIFICATION HEAD
    model.classifier = nn.Sequential(
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES)
    )
    
    model.load_state_dict(
        torch.load(
            "models/mobilenetv2_smartvision.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model


# EffcientNetB0
@st.cache_resource
def load_custom_EffcientNet():
    model=models.efficientnet_b0(pretrained=False)
    
    #  CUSTOM CLASSIFICATION HEAD
    model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
    )
    
    model.load_state_dict(
        torch.load(
            "models/EfficientNetB0_smartvision.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model

# Image preprocessing

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ------------------------------------------------------------------------------------------------------------------------------------

if page == "🖼️ Image Classification":
    st.subheader("🖼️ Image Classification (Custom Trained CNN Models)")
    st.markdown("""
    This page performs **single-object image classification** using multiple
    **custom-trained CNN models**.  
    Predictions from each model are shown **side-by-side** for comparison.
    """)

    uploaded_file = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### 📷 Uploaded Image")
        st.image(image, width=300)

        input_tensor = preprocess(image).unsqueeze(0)

        # Load all models
        models_dict = {
            "🧠 VGG16": load_custom_vgg16(),
            "🧠 ResNet50": load_custom_restnet50(),
            "🧠 MobileNetV2": load_custom_mobilenetv2(),
            "🧠 EfficientNet-B0": load_custom_EffcientNet()
        }

        st.markdown("---")
        st.markdown("### 🔍 Model Predictions (Top-5)")

        cols = st.columns(4)

        for col, (model_name, model) in zip(cols, models_dict.items()):
            with col:
                st.markdown(f"#### {model_name}")

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)

                top_probs, top_idxs = torch.topk(
                    probs, min(5, len(Classes))
                )

                for i in range(len(top_idxs)):
                    class_name = Classes[top_idxs[i].item()]
                    confidence = top_probs[i].item()

                    st.write(
                        f"**{i+1}. {class_name}** — {confidence*100:.2f}%"
                    )
                    st.progress(float(confidence))

    else:
        st.info("⬆️ Upload an image to classify.")




import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import streamlit as st


@st.cache_resource
def load_yolo_model():
    return YOLO("best (1).pt")   # path to my already trained model

yolo_model = load_yolo_model()

#-------------------------------------------------------------------------------------------------------------------------------------

if page == "📦 Object Detection":
    st.subheader("🎯 Object Detection using YOLO")
    st.markdown("""
    Upload an image to detect **multiple objects** using a custom-trained YOLO model.
    Bounding boxes, class labels, and confidence scores will be displayed.
    """)

    st.markdown("---")

    # Confidence threshold slider
    conf_threshold = st.slider(
        "🔧 Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    uploaded_file = st.file_uploader(
        "📤 Upload an Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.markdown("### 📷 Uploaded Image")
        st.image(image, width=350)

        st.markdown("---")
        st.markdown("### 🔍 Detection Results")

        # YOLO inference
        results = yolo_model.predict(
            source=img_array,
            conf=conf_threshold,
            save=False
        )

        annotated_img = img_array.copy()

        detections_found = False

        for r in results:
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:
                    detections_found = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label text
                    text = f"{label} {conf*100:.2f}%"
                    cv2.putText(
                        annotated_img,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        if detections_found:
            st.image(
                annotated_img,
                caption="YOLO Detection Output",
                use_column_width=True
            )
        else:
            st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")

    else:
        st.info("⬆️ Upload an image to start object detection.")




# This dashboard compares multiple CNN architectures based on accuracy and inference speed.
# While deeper models like VGG16 perform well during training, lightweight models such as
# MobileNetV2 and EfficientNetB0 offer faster inference, making them suitable for real-time applications.


#----------------------------------Model Performance---------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------- MODEL METRICS DATA ----------------
data = {
    "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
    "Train Accuracy": [0.877, 0.6815, 0.51, 0.5297],
    "Val Accuracy": [0.6345, 0.6855, 0.54, 0.56],
    "Test Accuracy": [0.633, 0.593, 0.579, 0.543],
    "Speed": [8.9, 0.5, 13.0, 12.6]  # higher = faster
}

df = pd.DataFrame(data)

# ---------------- PAGE 4: MODEL PERFORMANCE ----------------
if page == "📊 Model Performance":
    st.subheader("📊 Model Performance Dashboard")
    st.markdown("""
    This section presents a **comparative analysis** of different CNN models used in SmartVision AI.
    It highlights **training, validation, and test accuracy**, along with **relative inference speed**.
    """)

    st.markdown("---")

    # ---------------- MODEL METRICS TABLE ----------------
    st.markdown("### 📋 Model Comparison Table")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # ---------------- ACCURACY COMPARISON ----------------
    st.markdown("### 📈 Accuracy Comparison (Train / Validation / Test)")

    acc_df = df.melt(
        id_vars="Model",
        value_vars=["Train Accuracy", "Val Accuracy", "Test Accuracy"],
        var_name="Dataset",
        value_name="Accuracy"
    )

    fig1, ax1 = plt.subplots()
    sns.barplot(
        data=acc_df,
        x="Model",
        y="Accuracy",
        hue="Dataset",
        ax=ax1
    )
    ax1.set_ylim(0, 1)
    ax1.set_title("Accuracy Comparison Across Models")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Model")

    st.pyplot(fig1)

    st.markdown("---")

    # ---------------- INFERENCE SPEED COMPARISON ----------------
    st.markdown("### ⚡ Inference Speed Comparison")

    fig2, ax2 = plt.subplots()
    sns.barplot(
        data=df,
        x="Model",
        y="Speed",
        ax=ax2
    )
    ax2.set_title("Relative Inference Speed (Higher is Faster)")
    ax2.set_ylabel("Speed Score")
    ax2.set_xlabel("Model")

    st.pyplot(fig2)

    st.markdown("---")

    # ---------------- PERFORMANCE INSIGHTS ----------------
    st.markdown("### 🧠 Key Observations")
    st.markdown("""
    - **VGG16** shows strong training accuracy but noticeable generalization gap  
    - **ResNet50** provides better validation stability  
    - **MobileNetV2** and **EfficientNetB0** trade accuracy for faster inference  
    - Lightweight models are suitable for **real-time or edge deployment**
    """)



#----------------------------------------------------Live Camera Detection----------------------------------------------------------------------------------
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO

#Loading the pretrained model from YOLO

@st.cache_resource
def load_pretrained_yolo():
    return YOLO("yolov8n.pt")   # pretrained model

yolo_model_live = load_pretrained_yolo()


if page == "📸 Live Webcam Detection":
    st.subheader("📸 Live Camera Detection (Lightweight Mode)")
    
    # 0.5 → show only detections above 50% if 0.1 then show only detections above 10%
    conf_thres = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    run = st.checkbox("▶ Start Camera")

    FRAME_WINDOW = st.image([])
    fps_text = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # using this for faster optimization
        frame_skip = 3   # 🔥 process 1 frame out of 3
        frame_count = 0
        prev_time = time.time()

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames
            if frame_count % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = yolo_model_live.predict(
                frame_rgb,
                conf=conf_thres,
                imgsz=416,       # 🔥 smaller image
                verbose=False
            )

            annotated_frame = results[0].plot()

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            fps_text.markdown(f"⚡ FPS: {fps:.1f}")

            FRAME_WINDOW.image(
                annotated_frame,
                channels="RGB",
                use_column_width=True
            )

            time.sleep(0.03)   # 🔥 CPU cooldown

        cap.release()



if page == "ℹ️ About":
    st.subheader("📘 About SmartVision AI")
    st.markdown("---")

    # ---------------- PROJECT OVERVIEW ----------------
    st.markdown("## 🧠 Project Overview")
    st.markdown("""
    **SmartVision AI** is an end-to-end **computer vision system** designed to perform  
    **image classification**, **object detection**, and **real-time inference** using
    state-of-the-art deep learning models.

    The project demonstrates the complete AI lifecycle:
    **dataset preparation → model training → optimized inference → deployment using Streamlit**.
    """)

    # ---------------- DATASET INFO ----------------
    st.markdown("## 📂 Dataset Information")
    st.markdown("""
    - **Image Classification Dataset**
        - Domain-specific dataset with **25 object classes**
        - Preprocessed and augmented for robustness
        - Split into **Train / Validation / Test** sets

    - **Object Detection Dataset**
        - General object detection using **COCO dataset**
        - 80 commonly occurring object classes
        - Bounding-box annotated images
    """)

    # ---------------- MODEL ARCHITECTURES ----------------
    st.markdown("## 🏗️ Model Architectures Used")
    st.markdown("""
    ### 🔹 Image Classification Models
    - **VGG16 (Custom Trained)**
        - Modified fully connected layers
        - High accuracy on domain-specific data

    - **ResNet50**
        - Residual connections for deeper learning
        - Strong generalization capability

    - **MobileNetV2**
        - Lightweight architecture
        - Optimized for speed and mobile devices

    - **EfficientNet-B0**
        - Balanced accuracy and efficiency
        - Compound scaling technique

    ### 🔹 Object Detection Model
    - **YOLOv8 (Pretrained)**
        - Real-time object detection
        - Single-stage detector
        - Optimized for speed and accuracy
    """)

    # ---------------- TECH STACK ----------------
    st.markdown("## 🛠️ Technical Stack")
    st.markdown("""
    **Programming Language**
    - Python 🐍

    **Deep Learning & Vision**
    - PyTorch
    - Torchvision
    - Ultralytics YOLOv8
    - OpenCV

    **Data Processing & Visualization**
    - NumPy
    - Pandas
    - Matplotlib
    - Seaborn

    **Web & Deployment**
    - Streamlit
    - VS Code
    - Git & GitHub
    """)

    # ---------------- OPTIMIZATION ----------------
    st.markdown("## ⚡ Performance Optimization Techniques")
    st.markdown("""
    - Model quantization (where applicable)
    - Frame skipping for real-time inference
    - Resolution scaling for faster detection
    - CPU-optimized inference pipeline
    - Streamlit resource caching
    """)

    # ---------------- DEVELOPER INFO ----------------
    st.markdown("## 👨‍💻 Developer Information")
    st.markdown("""
    **Developer:** Rahul Kumar  
    **Degree:** B.Tech in Information Technology  
    **Institution:** IIEST Shibpur  

    **Core Interests:**
    - Computer Vision
    - Deep Learning
    - Full Stack Development
    - AI Model Deployment

    **Project Goal:**
    To build scalable, efficient, and production-ready AI systems
    with real-world deployment considerations.
    """)

    # ---------------------------------------FOOTER --------------------------------------------------------------------------------------
    st.markdown("---")
    st.info("🚀 SmartVision AI — Bridging Deep Learning Research with Real-World Applications")


#-------------------Footer Part in sidebar----------------------------------------------------------------------------------------------

import streamlit as st

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 SmartVision AI")

col1, col2, col3 = st.sidebar.columns(3)

with col1:
    st.sidebar.markdown(
        "[🌐 GitHub](https://github.com/rahul-tech-kumar/SmartVision-AI---Intelligent-Multi-Class-Object-Recognition-System)",
        unsafe_allow_html=True
    )

with col2:
    st.sidebar.markdown(
        "[💼 LinkedIn](https://www.linkedin.com/in/rahul-kumar-173546228/)",
        unsafe_allow_html=True
    )

with col3:
    st.sidebar.markdown(
        "[✉️ Email](mailto:rahulkumar11062003@gmail.com)",
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="text-align:center; font-size:12px; color:gray;">
        🚀 Built with Streamlit & PyTorch<br>
        © 2025 SmartVision AI
    </div>
    """,
    unsafe_allow_html=True
)
