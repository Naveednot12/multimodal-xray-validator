# 🫁 Multimodal Chest X-ray Report Validator

A deep learning system that validates whether a chest X-ray image and its radiology report are **consistent** with each other. Built using DenseNet121 for image feature extraction and BioClinicalBERT for text understanding, fused together with an MLP similarity model.

---

## 📌 Overview

In clinical settings, radiology reports can sometimes be mismatched with the wrong X-ray, or a report may be incorrectly written. This system acts as an **automated quality control tool** — it takes a chest X-ray and a radiology report as input, and outputs a consistency score indicating whether they match.

| Score | Prediction |
|-------|------------|
| > 0.59 | ✅ CONSISTENT — Report matches the X-ray |
| ≤ 0.59 | ❌ INCONSISTENT — Report does not match the X-ray |

---

## 🏗️ Architecture

```
MIMIC-CXR Dataset
       |
  _____|_____________________
  |                         |
  ↓                         ↓
X-ray Images         Radiology Reports
  |                         |
  ↓                         ↓
DenseNet121          BioClinicalBERT
(torchxrayvision)    (emilyalsentzer/Bio_ClinicalBERT)
  |                         |
  ↓                         ↓
1024 features          768 features
  |_________________________|
             |
             ↓
     Feature Fusion (1792-dim)
             |
             ↓
      StandardScaler
             |
             ↓
    MLP Similarity Model
    (1792 → 768 → 256 → 1)
             |
             ↓
      Consistency Score
             |
             ↓
      Streamlit Web App
```

---

## 🧠 Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Image Encoder | DenseNet121 (torchxrayvision) | Extracts 1024-dim visual features from chest X-rays |
| Text Encoder | BioClinicalBERT | Extracts 768-dim semantic features from radiology reports |
| Similarity Model | Custom MLP | Learns to compare fused image+text features |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~77% |
| Precision | ~78% |
| Recall | ~76% |
| Optimal Threshold | 0.59 |

---

## 📁 Project Structure

```
chest-xray-report-validator/
│
├── frontend/
│   └── app.py                              ← Streamlit web app
│
├── image_feature_extraction_densenet.py    ← Step 1: Extract image features
├── text_feature_extraction.py              ← Step 2: Extract text features
├── create_balanced_dataset_smart.py        ← Step 3: Build training pairs
├── fusion_densenet.py                      ← Step 4: Fuse features
├── train_similarity_densenet_smart_improved.py  ← Step 5: Train model
├── image_preprocessing.py                 ← Image preprocessing utilities
├── text_preprocessing.py                  ← Text preprocessing utilities
│
├── .gitignore
└── README.md
```

> **Note:** The `archive/` folder containing model weights (`.pth`), features (`.npy`), scaler (`.pkl`), and the MIMIC-CXR dataset are not included in this repository due to file size and data licensing restrictions.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/chest-xray-report-validator.git
cd chest-xray-report-validator

# Install dependencies
pip install streamlit torch torchvision
pip install transformers torchxrayvision
pip install scikit-learn joblib pandas numpy Pillow
```

---

## 🚀 Running the Pipeline (Training from Scratch)

Run the scripts in this order:

```bash
# Step 1 - Extract image features from MIMIC-CXR
python image_feature_extraction_densenet.py

# Step 2 - Extract text features from radiology reports
python text_feature_extraction.py

# Step 3 - Create balanced training dataset
python create_balanced_dataset_smart.py

# Step 4 - Fuse image and text features
python fusion_densenet.py

# Step 5 - Train the similarity model
python train_similarity_densenet_smart_improved.py
```

---

## 🖥️ Running the Web App

```bash
cd frontend
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📱 How to Use the App

1. **Upload** a chest X-ray image (JPG/PNG, max 5MB)
2. **Provide** the radiology report using one of three options:
   - 📁 Upload a `.txt` file containing the report
   - 🔍 Enter a Study ID to auto-fetch from the dataset
   - ✏️ Paste the report text directly
3. **Click** `Check Consistency`
4. **View** the Consistency Score and prediction result

---

## 📦 Dataset

This project uses the **MIMIC-CXR** dataset:
- 200,000+ chest X-ray images with radiology reports
- Publicly available at: [physionet.org/content/mimic-cxr](https://physionet.org/content/mimic-cxr/)
- Requires credentialed access via PhysioNet

---

## ⚠️ Limitations

- Model trained exclusively on MIMIC-CXR data — performance may drop on X-rays from different hospitals (domain gap)
- Expects short, clean, lowercased report text matching the training format
- Negative pairs distinguished primarily by pneumonia keyword — may not generalize to all conditions
- Not intended for real clinical deployment

---

## 🔮 Future Work

- Train on multiple hospital datasets to reduce domain gap
- Use cross-attention fusion instead of simple concatenation
- Implement harder negative mining strategies
- Add auto-cleaning of raw radiology reports

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

- **Python 3.10**
- **PyTorch** — deep learning framework
- **TorchXRayVision** — pretrained medical imaging models
- **HuggingFace Transformers** — BioClinicalBERT
- **Streamlit** — web app framework
- **scikit-learn** — StandardScaler, metrics
- **NumPy / Pandas** — data processing

---

## 📄 License

This project is for academic purposes only. The MIMIC-CXR dataset is subject to its own licensing terms via PhysioNet.
