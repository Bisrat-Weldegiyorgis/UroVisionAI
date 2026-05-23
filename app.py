# =========================================================
# UroVisionAI — Premium Medical Dashboard
# Developed by Bisrat Weldegiyorgis
# =========================================================

import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="UroVisionAI",
    page_icon="🧪",
    layout="wide"
)

# =========================================================
# PREMIUM CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #20458f 0%, #0d2b66 35%, #071736 100%);
    color: white;
}

/* WARNING */
.warning-box {
    background: linear-gradient(to right, rgba(250,204,21,0.18), rgba(250,204,21,0.08));
    border: 1px solid rgba(250,204,21,0.25);
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    font-size: 18px;
    font-weight: 700;
    color: #fde68a;
    margin-bottom: 18px;
}

/* PANEL */
.panel {
    background: linear-gradient(to bottom, #f8fafc, #e5e7eb);
    border-radius: 14px;
    border: 2px solid rgba(255,255,255,0.18);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}

.panel-header {
    background: linear-gradient(to bottom, #1e3a8a, #0d2b66);
    padding: 18px 26px;
    font-size: 26px;
    font-weight: 800;
    color: white;
}

.panel-body {
    padding: 20px;
}

/* METRICS */
.metric-row {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 12px;
    font-size: 22px;
    font-weight: 700;
    color: #1e293b;
}

/* UPLOAD */
.upload-box {
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    background: white;
    margin-bottom: 18px;
}

.upload-text {
    color: #1e3a8a;
    font-size: 20px;
    font-weight: 700;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    padding: 14px;
    border-radius: 10px;
    border: none;
    color: white;
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(to bottom, #1d4ed8, #0d2b66);
}

.stButton > button:hover {
    background: linear-gradient(to bottom, #2563eb, #1d4ed8);
}

/* CLINICAL */
.clinical-box {
    background: white;
    border-radius: 12px;
    padding: 18px;
    color: #0f172a;
    border: 1px solid #cbd5e1;
}

/* RISK */
.low { color: #16a34a; font-weight: 900; }
.medium { color: #d97706; font-weight: 900; }
.high { color: #dc2626; font-weight: 900; }

/* FOOTER */
.footer {
    text-align: center;
    color: #dbeafe;
    font-size: 16px;
    margin-top: 20px;
}

/* =========================================================
   BIG SHINY TITLE (NEW)
========================================================= */
.title {
    text-align: center;
    font-size: 78px;
    font-weight: 1000;
    letter-spacing: -2px;
    margin-top: 10px;
    background: linear-gradient(90deg, #ffffff, #60a5fa, #facc15, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 25px rgba(96,165,250,0.6);
    animation: glow 2.5s ease-in-out infinite alternate;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #dbeafe;
    margin-bottom: 20px;
    letter-spacing: 2px;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 10px rgba(96,165,250,0.4)); }
    to   { filter: drop-shadow(0 0 28px rgba(250,204,21,0.6)); }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE (BIG SHINY)
# =========================================================
st.markdown("""
<div class="title">
    🧪 UroVision<span style="color:#facc15;">AI</span>
</div>
<div class="subtitle">
    AI-Powered Clinical Urine Intelligence System
</div>
""", unsafe_allow_html=True)

# WARNING
st.markdown("""
<div class="warning-box">
⚠️ Shake sample tube, wait 30 minutes before analysis
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
with open("dataset/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cpu")
model = CNN(len(le.classes_))

model.load_state_dict(torch.load("dataset/urine_cnn.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =========================================================
# REPORT ENGINE
# =========================================================
def generate_report(color, foam, clarity):

    if color in ["yellow", "dark_yellow"]:
        return "Mild Dehydration", "LOW", "Concentrated urine detected.", "Drink more water."

    elif color in ["red", "red_pink"]:
        return "Possible Hematuria", "HIGH", "Blood detected in urine.", "Seek medical attention."

    elif clarity == "cloudy":
        return "Possible UTI", "MEDIUM", "Cloudiness detected.", "Do urinalysis."

    return "Normal Urine Profile", "LOW", "Healthy urine appearance.", "Maintain hydration."

# =========================================================
# PREDICT
# =========================================================
def predict(image):

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

        label = le.inverse_transform(pred.cpu().numpy())[0]
        confidence = round(confidence.item() * 100, 2)

        parts = label.split("_")

        if len(parts) >= 3:
            color = parts[0]
            foam = parts[1]
            clarity = "_".join(parts[2:])
        else:
            color = foam = clarity = "unknown"

        diagnosis, risk, findings, recommendation = generate_report(color, foam, clarity)

        return {
            "color": color.title(),
            "foam": foam.title(),
            "clarity": clarity.title(),
            "confidence": confidence,
            "diagnosis": diagnosis,
            "risk": risk,
            "findings": findings,
            "recommendation": recommendation
        }

# =========================================================
# UI
# =========================================================
left_col, right_col = st.columns([1, 2])
image = None

with left_col:

    st.markdown("""
    <div class="panel">
        <div class="panel-header">Upload Sample</div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    mode = st.radio("Select Input Method", ["Upload Image", "Capture via Camera"])

    if mode == "Upload Image":
        file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
        if file:
            image = Image.open(file)
            st.image(image)

    else:
        file = st.camera_input("Capture image")
        if file:
            image = Image.open(file)
            st.image(image)

    analyze = st.button("🔍 Analyze Sample")

    st.markdown("</div></div>", unsafe_allow_html=True)

with right_col:

    if image is not None and analyze:

        result = predict(image)

        st.markdown("""
        <div class="panel">
            <div class="panel-header">Prediction Results</div>
            <div class="panel-body">
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='metric-row'>Color: {result['color']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-row'>Foam: {result['foam']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-row'>Clarity: {result['clarity']}</div>", unsafe_allow_html=True)

        st.progress(result["confidence"] / 100)
        st.markdown(f"## {result['confidence']}%")

        st.image(image)

        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <div class="panel-header">Clinical Insights</div>
            <div class="panel-body">
        """, unsafe_allow_html=True)

        risk_class = result["risk"].lower()

        st.markdown(f"""
        <div class="clinical-box">
        <h2>Risk: <span class="{risk_class}">{result['risk']}</span></h2>
        <h3>Diagnosis</h3><p>{result['diagnosis']}</p>
        <h3>Findings</h3><p>{result['findings']}</p>
        <h3>Recommendation</h3><p>{result['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
© 2026 UroVisionAI — Bisrat Weldegiyorgis
</div>
""", unsafe_allow_html=True)
