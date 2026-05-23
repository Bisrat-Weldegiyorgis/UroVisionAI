import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pickle

# -----------------------
# LOAD LABEL ENCODER
# -----------------------
with open("dataset/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------
# CNN MODEL
# -----------------------
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

# -----------------------
# LOAD MODEL
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(len(le.classes_))
model.load_state_dict(torch.load("dataset/urine_cnn.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------
# IMAGE TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------
# LABEL DECODER
# -----------------------
def decode_label(label):
    parts = label.split("_")
    color = parts[0]
    foam = parts[1]
    clarity = "_".join(parts[2:])
    return color, foam, clarity

# -----------------------
# FULL MEDICAL REPORT
# -----------------------
def full_medical_report(color, foam, clarity, confidence):

    # NORMAL CASE
    if color == "pale_yellow" and foam == "none" and clarity == "clear":
        diagnosis = "Normal Urine Profile"
        risk = "LOW"
        findings = (
            "Pale yellow urine with clear appearance and no foam detected.\n"
            "This indicates normal hydration status and healthy renal function."
        )

    # DEHYDRATION
    elif color in ["yellow", "dark_yellow"] and foam == "none":
        diagnosis = "Mild to Moderate Dehydration"
        risk = "LOW"
        findings = (
            "Concentrated urine suggests reduced fluid intake.\n"
            "No abnormal foam or turbidity detected."
        )

    # HEMATURIA
    elif color == "red_pink":
        diagnosis = "Hematuria Suspected"
        risk = "HIGH"
        findings = (
            "Red/pink urine indicates possible blood presence.\n"
            "Possible causes: infection, stones, trauma, renal disease."
        )

    # LIVER / MUSCLE
    elif color == "brown":
        diagnosis = "Possible Hepatic or Muscle Disorder"
        risk = "HIGH"
        findings = (
            "Brown urine may indicate bilirubin or myoglobin presence.\n"
            "Immediate medical evaluation recommended."
        )

    # PROTEINURIA
    elif "moderate" in foam or "foam" in foam:
        diagnosis = "Possible Proteinuria"
        risk = "MEDIUM"
        findings = (
            "Foam suggests possible protein leakage in urine.\n"
            "May indicate early kidney dysfunction."
        )

    # INFECTION
    elif "cloudy" in clarity:
        diagnosis = "Possible Urinary Tract Infection"
        risk = "MEDIUM"
        findings = (
            "Cloudy urine suggests infection or sediment presence.\n"
            "Further urinalysis recommended."
        )

    # DEFAULT
    else:
        diagnosis = "Non-specific Finding"
        risk = "LOW"
        findings = (
            "No strong pathological indicators detected.\n"
            "Clinical correlation recommended."
        )

    # FINAL REPORT FORMAT
    return f"""
🧪 URINALYSIS REPORT (AI-ASSISTED SYSTEM)
===========================================

📌 DIAGNOSIS: {diagnosis}

📊 CONFIDENCE: {round(confidence * 100, 2)} %

🔬 FINDINGS:
{findings}

⚠️ CLINICAL RISK LEVEL: {risk}

🧠 MEDICAL NOTE:
This AI system provides decision support only and does not replace laboratory testing or clinical diagnosis.

===========================================
"""

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

        label = le.inverse_transform(pred.cpu().numpy())[0]
        confidence = confidence.item()

        color, foam, clarity = decode_label(label)

        report = full_medical_report(color, foam, clarity, confidence)

        print(report)

# -----------------------
# RUN TEST
# -----------------------
predict("urine specimen/U001.jpg")
