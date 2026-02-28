import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ===== PATHS =====
CSV_PATH = r"D:\Final yr Project Dataset\archive\paired_mimic_cxr_with_text.csv"
OUT_FEATURES = r"D:\Final yr Project Dataset\archive\text_features.npy"
OUT_IDS = r"D:\Final yr Project Dataset\archive\text_ids.npy"

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== LOAD DATA =====
df = pd.read_csv(CSV_PATH)
texts = df["cleaned_report"].fillna("").tolist()
study_ids = df["study_id"].tolist()

print("Total reports:", len(texts))

# ===== LOAD CLINICALBERT =====
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    use_safetensors=True
).to(device)

model.eval()

# ===== FEATURE EXTRACTION =====
features = []

with torch.no_grad():
    for i, text in enumerate(tqdm(texts)):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        features.append(cls_embedding.cpu().numpy().squeeze())

# ===== SAVE =====
np.save(OUT_FEATURES, np.array(features))
np.save(OUT_IDS, np.array(study_ids))

print("🎉 Text feature extraction completed")
print("Saved:", OUT_FEATURES)
