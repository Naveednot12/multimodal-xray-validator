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

st.set_page_config(
    page_title="Multimodal Chest X-ray Report Validator",
    page_icon="XR",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Source+Sans+3:wght@400;600&display=swap');

    :root {
        --bg: #f6f9ff;
        --surface: #ffffff;
        --surface-soft: #f2f7ff;
        --text: #0f172a;
        --muted: #5b6b86;
        --primary: #1d4ed8;
        --line: #c7d8f5;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% -12%, #d8e9ff 0%, transparent 30%),
            radial-gradient(circle at 90% 4%, #e5efff 0%, transparent 30%),
            var(--bg);
        color: var(--text);
        font-family: 'Source Sans 3', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1120px;
    }

    .topbar {
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 0.95rem 1rem 0.85rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.25rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.55rem;
        color: #0f172a;
        text-align: center;
    }

    .brand-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--primary), #3b82f6);
        box-shadow: 0 0 14px rgba(59, 130, 246, 0.45);
    }

    .topbar-note {
        color: #334155;
        font-size: 0.92rem;
        text-align: center;
    }

    .hero {
        border: 1px solid var(--line);
        background: linear-gradient(135deg, #ffffff 0%, #f0f6ff 100%);
        border-radius: 16px;
        padding: 1.2rem 1.25rem;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-family: 'Sora', sans-serif;
        font-size: 1.75rem;
        line-height: 1.2;
    }

    .hero p {
        margin: 0.5rem 0 0 0;
        color: var(--muted);
        max-width: 760px;
    }

    .tech-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.25rem 0 1rem 0;
    }

    .tech-chip {
        background: #eef4ff;
        color: #1e3a8a;
        border: 1px solid #c7d8f5;
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .panel {
        border: 1px solid var(--line);
        background: linear-gradient(180deg, var(--surface) 0%, var(--surface-soft) 100%);
        border-radius: 14px;
        padding: 1rem;
        min-height: 160px;
    }

    .panel h3 {
        margin: 0 0 0.55rem 0;
        font-family: 'Sora', sans-serif;
        font-size: 1rem;
    }

    .muted {
        color: var(--muted);
        font-size: 0.95rem;
    }

    .readiness-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.4rem;
        margin-bottom: 0.65rem;
    }

    .readiness-item {
        border: 1px solid #c7d8f5;
        border-radius: 10px;
        background: #ffffff;
        padding: 0.55rem 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.93rem;
        color: #1f2937;
        min-height: 42px;
    }

    .readiness-item .state {
        font-weight: 700;
        font-size: 0.95rem;
    }

    .readiness-item.ready {
        background: #f0fdf4;
        border-color: #86efac;
    }

    .readiness-item.ready .state {
        color: #166534;
    }

    .readiness-item.not-ready {
        background: #fff7ed;
        border-color: #fdba74;
    }

    .readiness-item.not-ready .state {
        color: #9a3412;
    }

    .status-box {
        border: 1px solid #c7d8f5;
        border-radius: 12px;
        padding: 1rem;
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        text-align: center;
        margin-top: 1rem;
    }

    .status-box.consistent {
        border: 1px solid #86efac;
        background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%);
    }

    .status-box.inconsistent {
        border: 1px solid #fca5a5;
        background: linear-gradient(180deg, #fef2f2 0%, #fee2e2 100%);
    }

    .status-title {
        font-family: 'Sora', sans-serif;
        font-size: 0.9rem;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }

    .status-main {
        font-family: 'Sora', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
    }

    .status-box.consistent .status-main {
        color: #166534;
    }

    .status-box.inconsistent .status-main {
        color: #991b1b;
    }

    .reason-line {
        margin-top: 0.6rem;
        color: #334155;
        font-size: 0.95rem;
    }

    .timestamp-line {
        margin-top: 0.35rem;
        color: #64748b;
        font-size: 0.85rem;
    }

    div[data-testid="stFileUploader"] {
        margin-top: 0.9rem;
        margin-bottom: 0.85rem;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.35rem;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1px dashed #7aa2f8 !important;
        border-radius: 12px;
    }

    div[data-testid="stFileUploaderFile"],
    div[data-testid="stFileUploaderFile"] * {
        color: #0f172a !important;
    }

    .image-marker + div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzoneInstructions"] small,
    .report-marker + div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none;
    }

    div.stButton > button,
    div.stButton > button:hover,
    div.stButton > button:active,
    div.stButton > button:focus,
    div.stButton > button:focus-visible,
    div.stButton > button:disabled {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
        <div class="brand"><span class="brand-dot"></span>Multimodal Consistency Validator</div>
        <div class="topbar-note">DenseNet121 + BioClinicalBERT + MLP similarity pipeline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Validate if a Radiology Report Matches a Chest X-ray</h1>
        <p>
            Upload a chest X-ray and provide a report by file, Study ID, or pasted text.
            The underlying inference pipeline is unchanged and uses the trained multimodal model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tech-strip">
        <span class="tech-chip">DenseNet121</span>
        <span class="tech-chip">BioClinicalBERT</span>
        <span class="tech-chip">Early Fusion</span>
        <span class="tech-chip">MLP Similarity Model</span>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="panel"><h3>Image Input</h3><div class="muted">Upload a frontal chest X-ray image.</div>', unsafe_allow_html=True)
    st.markdown('<div class="image-marker"></div>', unsafe_allow_html=True)
    image_file = st.file_uploader("Upload image (JPG/PNG, max 5MB)",
                                   type=["jpg", "jpeg", "png"],
                                   label_visibility="collapsed")
    if image_file and image_file.size > 5 * 1024 * 1024:
        st.error("File exceeds 5MB limit.")
        image_file = None
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel"><h3>Report Input</h3><div class="muted">Upload a radiology report text file (.txt).</div>', unsafe_allow_html=True)
    st.markdown('<div class="report-marker"></div>', unsafe_allow_html=True)
    report_file = st.file_uploader("Upload .txt report (max 5MB)",
                                    type=["txt"],
                                    label_visibility="collapsed")
    if report_file and report_file.size > 5 * 1024 * 1024:
        st.error("File exceeds 5MB limit.")
        report_file = None
    st.markdown('</div>', unsafe_allow_html=True)

report_ready = bool(report_file is not None)

st.markdown("### Input Readiness")
st.markdown(
    f"""
    <div class="readiness-grid">
        <div class="readiness-item {'ready' if image_file is not None else 'not-ready'}">
            <span class="state">{'✓' if image_file is not None else '○'}</span>
            <span>{'Image file selected' if image_file is not None else 'Image file not selected'}</span>
        </div>
        <div class="readiness-item {'ready' if report_ready else 'not-ready'}">
            <span class="state">{'✓' if report_ready else '○'}</span>
            <span>{'Report source provided' if report_ready else 'Report source not provided'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if image_file or report_file:
    p1, p2 = st.columns(2, gap="large")
    with p1:
        if image_file:
            st.markdown("**Image Preview**")
            st.image(image_file, caption=image_file.name, use_container_width=True)
    with p2:
        if report_file is not None:
            st.markdown("**Report Preview**")
            report_preview = report_file.getvalue().decode("utf-8", errors="ignore").strip()
            st.text_area(
                "Uploaded report preview",
                value=report_preview[:600] if report_preview else "(Empty report file)",
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )

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
        st.warning("Please upload a chest X-ray image.")
    else:
        report_text = None
        source_label = ""

        if report_file is not None:
            report_text = report_file.read().decode("utf-8").strip()
            source_label = f"Using uploaded file: `{report_file.name}`"

        else:
            st.warning("Please upload a radiology report (.txt).")

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

            if score > 0.59:
                prediction_text = "CONSISTENT"
                reason_text = "Visual pattern and report narrative are aligned for the same clinical condition."
                prediction_class = "consistent"
            else:
                prediction_text = "INCONSISTENT"
                reason_text = "Visual findings and report narrative appear mismatched for the same clinical condition."
                prediction_class = "inconsistent"

            st.markdown(
                f"""
                <div class="status-box {prediction_class}">
                    <div class="status-title">Prediction</div>
                    <div class="status-main">{prediction_text}</div>
                    <div class="reason-line">Reason: {reason_text}</div>
                    <div class="timestamp-line">Generated at: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )