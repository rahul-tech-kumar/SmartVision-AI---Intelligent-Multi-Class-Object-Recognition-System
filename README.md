# 🧠 SmartVision AI  
### Intelligent Multi-Class Object Recognition System

SmartVision AI is an **end-to-end Computer Vision application** that performs  
**image classification**, **object detection**, and **real-time inference** using
state-of-the-art deep learning models.  
The project demonstrates the complete AI lifecycle — from **model training** to
**optimized deployment** using **Streamlit**.

---

## 🚀 Key Features

- 🖼️ **Image Classification**
  - Custom-trained deep learning models
  - Top-5 prediction display with confidence scores
  - Side-by-side comparison of multiple CNN architectures

- 📦 **Object Detection**
  - Pretrained YOLOv8 model for real-time object detection
  - Bounding boxes, class labels, and confidence scores
  - Adjustable confidence threshold

- 📸 **Live Webcam Detection (Optimized)**
  - Real-time detection using webcam
  - FPS monitoring and CPU-friendly optimizations
  - Frame skipping and resolution scaling

- 📊 **Model Performance Dashboard**
  - Accuracy comparison (Train / Validation / Test)
  - Inference speed analysis
  - Visual performance insights

- ⚡ **Optimized Inference**
  - Lightweight models for CPU execution
  - Streamlit caching for faster loading
  - Performance-focused design decisions

---

## 🏗️ Model Architectures Used

### 🔹 Image Classification
- **VGG16 (Custom Trained)**
- **ResNet50**
- **MobileNetV2**
- **EfficientNet-B0**

### 🔹 Object Detection
- **YOLOv8 (Pretrained on COCO Dataset)**

---

## 📂 Dataset Information

- **Image Classification Dataset**
  - Domain-specific dataset
  - 25 object classes
  - Train / Validation / Test split
  - Image preprocessing and augmentation applied

- **Object Detection Dataset**
  - COCO Dataset
  - 80 general-purpose object classes
  - Bounding box annotations

---

## 🛠️ Tech Stack

**Programming Language**
- Python 🐍

**Deep Learning & Computer Vision**
- PyTorch
- Torchvision
- Ultralytics YOLOv8
- OpenCV

**Data Analysis & Visualization**
- NumPy
- Pandas
- Matplotlib
- Seaborn

**Web & Deployment**
- Streamlit
- VS Code
- Git & GitHub

---

## ⚡ Performance Optimization Techniques

- Frame skipping for real-time webcam inference
- Reduced image resolution for faster detection
- Lightweight YOLOv8n model for CPU execution
- Streamlit resource caching
- Confidence-based filtering of predictions

---

## 📁 Project Structure
SmartVisionAI/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── Image.txt/ # Images, icons, logos
├── yolo.ipynb
└── smartvisionAI.ipynb(Downloading and training process of data)

## Deployment
- HuggingFace: https://huggingface.co/spaces/rahulkumar11062003/Smartvision-Ai


## Screenshots
<img width="1920" height="1080" alt="Screenshot (176)" src="https://github.com/user-attachments/assets/feb97730-b862-4504-ac37-bc733fe21aba" />

<img width="1920" height="1080" alt="Screenshot (178)" src="https://github.com/user-attachments/assets/60abc70b-3d2c-4aee-aaf2-53dade77d7e3" />
Demo Images

<img width="1920" height="1080" alt="Screenshot (177)" src="https://github.com/user-attachments/assets/5c03e4b4-eaa5-4eb8-942f-b3fae16db210" />
Detection

<img width="1253" height="825" alt="Screenshot 2025-12-14 at 08-17-06 SmartVision AI - Intelligent Multi-Class Object Recognition System" src="https://github.com/user-attachments/assets/158bdfca-c160-4968-a508-d3cd47878768" />




## 📌 Note on Model Files

-Due to size constraints, trained model weights (.pt, .pth) are not included
in this repository.

## 🎓 Academic & Practical Relevance

- This project was built to:

- Demonstrate practical Deep Learning & Computer Vision skills

- Showcase model deployment and optimization

- Serve as a portfolio project for interviews and evaluations

## 👨‍💻 Developer

Rahul Kumar
B.Tech in Information Technology
IIEST Shibpur
