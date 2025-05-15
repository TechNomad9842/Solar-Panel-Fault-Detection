import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Streamlit config ---
st.set_page_config(page_title="Solar Fault Detection", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 1rem;
    }
    h1 {
        color: #333333;
    }
    .css-1v0mbdj {
        padding: 2rem 3rem 3rem 3rem;
    }
    .stButton > button {
        background-color: #ffcc00;
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Class labels ---
class_names = ["Functional", "Faulty"]

# --- Title Section ---
st.title("☀️ Mini Project App: Solar Panel Fault Detection")
st.markdown("This app uses a **ResNet-50** model to classify solar panels as **Functional** or **Faulty** from EL images, and provides a **fault severity score**.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("resnet50_softmax_solar.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Prediction Function ---
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(probs, 1)
        severity_score = probs[0][1].item()
    return pred.item(), severity_score, probs.numpy()

# --- Sidebar Upload ---
st.sidebar.header("📤 Upload EL Image")
uploaded_file = st.sidebar.file_uploader("Choose an EL image of a solar panel", type=["jpg", "png", "jpeg"])

# --- Prediction Display ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    pred_class, severity_score, probs = predict(image)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** `{class_names[pred_class]}`")
    st.metric("⚠️ Fault Severity Score", f"{severity_score:.2f}")

    # --- Bar Chart for Class Probabilities ---
    fig, ax = plt.subplots()
    sns.barplot(x=class_names, y=probs[0], palette="viridis", ax=ax)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    st.pyplot(fig)

# --- Sidebar Evaluation ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Evaluation")
st.sidebar.write("✅ **Accuracy:** `83.05%`")
st.sidebar.write("✅ **ROC AUC:** `0.894`")

# --- Classification Report ---
with st.expander("📌 Classification Report"):
    st.text("""
              precision    recall  f1-score   support

           0       0.81      0.93      0.86       302
           1       0.88      0.70      0.78       223

    accuracy                           0.83       525
   macro avg       0.84      0.81      0.82       525
weighted avg       0.84      0.83      0.83       525
""")

# --- Confusion Matrix ---
with st.expander("🧩 Confusion Matrix"):
    cm = np.array([[280, 22], [67, 156]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# --- Footer ---
st.markdown("<br><hr><center>Made with ❤️ for Mini Project 2025</center>", unsafe_allow_html=True)
