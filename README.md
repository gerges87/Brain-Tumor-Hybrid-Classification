# 🧠 Brain Tumor MRI Classification using Hybrid PDSCNN & RRELM

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)

## 📌 Project Overview
This project presents an advanced, lightweight Hybrid Deep Learning pipeline for classifying Brain Tumor MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary, and No Tumor**. 

Instead of relying on heavy pre-trained models, this project builds a custom **Parallel Depthwise Separable CNN (PDSCNN)** integrated with an **Attention Mechanism** to extract high-level features. These features are then classified using an optimized **Ridge Regression Extreme Learning Machine (RRELM)**, achieving high accuracy with a fraction of the computational cost compared to traditional architectures.

## ✨ Key Features & Architecture
1. **Advanced Image Pre-processing:** Utilized **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance MRI contrast, effectively highlighting tumor boundaries for better feature detection.
2. **Feature Extractor (PDSCNN + Attention):** A custom lightweight CNN utilizing parallel branches (3x3 and 5x5 filters) and **Squeeze-and-Excitation (SE) blocks** to focus on vital tumor tissues while ignoring non-informative background noise.
3. **Hybrid Classifier (RRELM):** Replaced the traditional Softmax layer with an Extreme Learning Machine optimized via Ridge Regression and MinMaxScaler, hyper-tuned using Automated Grid Search.
4. **Explainable AI (XAI):** Implemented **Grad-CAM** to generate heatmaps, visually proving that the model's predictions are grounded in actual tumor locations, ensuring medical interpretability and trustworthiness.

## 📊 Performance & Results
After rigorous Grid Search hyperparameter tuning (Neurons: **6000**, Alpha: **2.0**), the model achieved:
* **Validation Accuracy:** `94.17%`
* **Test Accuracy:** `89.12%`
* **Medical Reliability:** Achieved `100% Recall` for the **No-Tumor** class and `99% Recall` for **Pituitary** tumors, minimizing false negatives in critical diagnoses.

## 🛠️ Files Included
* `brain-tumor-classification-v2.ipynb`: The complete End-to-End code pipeline.
* `PDSCNN_FeatureExtractor_V2.h5`: The trained lightweight feature extractor model.
* `RRELM_Hybrid_Classifier_V2.pkl`: The trained fast classifier (Ridge + Scaler).
* `class_labels.json`: Dictionary mapping numerical model outputs to clinical class names.

## 🚀 How to Run (Deployment Ready)
To use this hybrid model for inference on a new MRI scan:

```python
import joblib
import json
import tensorflow as tf

# 1. Load Components
loaded_cnn = tf.keras.models.load_model('PDSCNN_FeatureExtractor_V2.h5')
loaded_rrelm = joblib.load('RRELM_Hybrid_Classifier_V2.pkl')
with open('class_labels.json', 'r') as f:
    labels = json.load(f)

# 2. Prediction Pipeline (After Preprocessing)
# features = loaded_cnn.predict(preprocessed_image)
# prediction = loaded_rrelm.predict(features)
# print(f"Diagnosis: {labels[str(prediction[0])]}")

👨‍💻 Author
Gerges Sobhy AI Engineer Passionate about applying Deep Learning and Computer Vision to real-world healthcare and medical imaging challenges.
