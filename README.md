# 🧪 UroVisionAI

UroVisionAI is a deep learning-based web application for analyzing urine images. It uses a trained PyTorch CNN model and provides a simple Streamlit interface for real-time predictions and clinical-style insights.

---

## 🚀 Features

- 🧠 CNN-based urine image classification
- 🖼️ Upload or capture images via camera
- ⚡ Fast inference using PyTorch
- 📊 Prediction of:
  - Color
  - Foam
  - Clarity
- 🏥 Clinical-style report generation:
  - Diagnosis
  - Risk level
  - Recommendations
- 🌐 Interactive Streamlit web interface

---

## 📁 Project Structure

UroVisionAI/
 - app.py                  # Streamlit web application
 - dataset/
    - label_encoder.pkl    # Label encoder for classes
    - urine_cnn.pth        # Trained PyTorch model
 - requirements.txt        # Project dependencies
 - README.md               # Project documentation

---

## ⚙️ Installation

```bash id="p3m9aa"
# 1. Clone the repository
git clone https://github.com/your-username/UroVisionAI.git

# 2. Move into project folder
cd UroVisionAI

# 3. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt



# 5. Run the app
streamlit run app.py
