import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchxrayvision as xrv
import joblib
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    img_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    img_model = img_model.features
    img_model.to(device)
    img_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    txt_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    txt_model.to(device)
    txt_model.eval()

    class SimilarityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1792, 768), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(768, 256),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = SimilarityModel().to(device)
    model.load_state_dict(
        torch.load("../archive/similarity_model_smart_improved.pth", map_location=device)
    )
    model.eval()

    scaler = joblib.load("../archive/scaler.pkl")
    return img_model, tokenizer, txt_model, model, scaler


@st.cache_data
def load_reports():
    df = pd.read_csv("../archive/paired_mimic_cxr_with_text.csv")
    df["study_id"] = df["study_id"].astype(str)
    df["cleaned_report"] = df["cleaned_report"].fillna("").str.lower()
    return df


img_model, tokenizer, txt_model, model, scaler = load_models()
reports_df = load_reports()

# -----------------------------
# UI
# -----------------------------
st.title("Multimodal Chest X-ray Report Validator")
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>Input Section</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("**🩻 Chest X-ray Image**")
    image_file = st.file_uploader("Upload image (JPG/PNG, max 5MB)",
                                   type=["jpg", "jpeg", "png"],
                                   label_visibility="collapsed")
    if image_file and image_file.size > 5 * 1024 * 1024:
        st.error("❌ File exceeds 5MB limit.")
        image_file = None

with col2:
    st.markdown("**📋 Radiology Report**")
    report_tab1, report_tab2, report_tab3 = st.tabs(["📁 Upload .txt", "🔍 Study ID", "✏️ Paste Text"])

    with report_tab1:
        report_file = st.file_uploader("Upload .txt report (max 5MB)",
                                        type=["txt"],
                                        label_visibility="collapsed")
        if report_file and report_file.size > 5 * 1024 * 1024:
            st.error("❌ File exceeds 5MB limit.")
            report_file = None

    with report_tab2:
        study_id_input = st.text_input("Study ID",
                                        placeholder="e.g. s55812956",
                                        label_visibility="collapsed")

    with report_tab3:
        manual_report = st.text_area("Paste report", height=150,
                                      placeholder="e.g. no acute cardiopulmonary abnormality",
                                      label_visibility="collapsed")

st.markdown("---")

# -----------------------------
# Feature extraction
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_image_feature(image):
    image = image.convert("L")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = img_model(img_tensor)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(1, -1)
    return feat.cpu().numpy().flatten()


def extract_text_feature(text):
    text = text.lower().strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = txt_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy().flatten()


if st.button("🔍 Check Consistency", use_container_width=True, type="primary"):

    if not image_file:
        st.warning("⚠️ Please upload a chest X-ray image.")
    else:
        report_text = None
        source_label = ""

        if manual_report.strip():
            report_text = manual_report.strip()
            source_label = f"✏️ Using pasted report"

        elif report_file is not None:
            report_text = report_file.read().decode("utf-8").strip()
            source_label = f"📁 Using uploaded file: `{report_file.name}`"

        elif study_id_input.strip():
            sid = study_id_input.strip()
            match = reports_df[reports_df["study_id"] == sid]
            if len(match) > 0:
                report_text = match.iloc[0]["cleaned_report"]
                source_label = f"🔍 Fetched report for Study ID: `{sid}`"
            else:
                st.error(f"❌ Study ID `{sid}` not found in dataset.")

        else:
            st.warning("⚠️ Please provide a report using one of the three options.")

        if report_text:
            st.caption(source_label)

            image = Image.open(image_file)
            img_feat = extract_image_feature(image)
            txt_feat = extract_text_feature(report_text)

            fused = np.concatenate([img_feat, txt_feat])
            fused_scaled = scaler.transform([fused])
            fused_tensor = torch.tensor(fused_scaled, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(fused_tensor)
                score = torch.sigmoid(logits).item()

            st.markdown("## 📊 Output Section")
            col_score, col_result = st.columns(2)

            with col_score:
                st.metric(label="Consistency Score", value=round(score, 4))
            with col_result:
                if score > 0.59:
                    st.success("✅ Prediction: CONSISTENT")
                else:
                    st.error("❌ Prediction: INCONSISTENT")