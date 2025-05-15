# ☀️ Solar Panel Fault Detection - Mini Project App

This is a **Streamlit web application** that detects faults in solar panel cells using **deep learning (ResNet-50)** on Electroluminescence (EL) images. The app also calculates a **fault severity score** based on model confidence.

---

## 🧠 Model Overview

- **Architecture**: ResNet-50 (PyTorch)
- **Output**: Binary classification (`Functional` or `Faulty`)
- **Extra**: Softmax-based **fault severity score** (0 to 1)
- **Dataset**: ELPV dataset (Electroluminescence Photovoltaic Images)

---

## 🧪 Model Evaluation

| Metric           | Value  |
| ---------------- | ------ |
| ✅ Accuracy      | 83.05% |
| ✅ ROC AUC       | 0.894  |
| ✅ Precision (0) | 0.81   |
| ✅ Recall (1)    | 0.70   |
