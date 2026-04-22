import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

print("Loading aligned dataset...")
aligned_df = pd.read_csv("archive/aligned_dataset_densenet.csv")

print("Loading DenseNet image features...")
image_features = np.load("archive/image_features_densenet.npy")

print("Loading text features...")
text_features = np.load("archive/text_features.npy")

print("Loading full report CSV...")
reports_df = pd.read_csv("archive/paired_mimic_cxr_with_text.csv")

# Lowercase text for safety
reports_df["cleaned_report"] = reports_df["cleaned_report"].fillna("").str.lower()

# Build study → text index map
study_to_text_index = {}
study_to_text_content = {}

for idx, row in reports_df.iterrows():
    study_id = str(row["study_id"])
    study_to_text_index[study_id] = idx
    study_to_text_content[study_id] = row["cleaned_report"]

print("Preparing positive samples...")
positive_pairs = []
labels = []

for _, row in aligned_df.iterrows():
    study_id = str(row["study_id"])
    img_idx = row["image_feature_index"]

    if study_id in study_to_text_index:
        txt_idx = study_to_text_index[study_id]
        positive_pairs.append((img_idx, txt_idx))
        labels.append(1)

print("Total positive samples:", len(positive_pairs))

print("Generating smarter negative samples...")

all_studies = list(study_to_text_index.keys())
negative_pairs = []

for img_idx, txt_idx in positive_pairs:
    original_study = None

    # Find study of this text
    for k, v in study_to_text_index.items():
        if v == txt_idx:
            original_study = k
            break

    original_text = study_to_text_content[original_study]

    # Determine keyword condition
    contains_pneumonia = "pneumonia" in original_text

    wrong_study = random.choice(all_studies)

    # Enforce opposite keyword condition
    while True:
        wrong_text = study_to_text_content[wrong_study]

        if contains_pneumonia and "pneumonia" not in wrong_text:
            break
        if not contains_pneumonia and "pneumonia" in wrong_text:
            break

        wrong_study = random.choice(all_studies)

    wrong_txt_idx = study_to_text_index[wrong_study]

    negative_pairs.append((img_idx, wrong_txt_idx))
    labels.append(0)

print("Total negative samples:", len(negative_pairs))

all_pairs = positive_pairs + negative_pairs
all_pairs, labels = shuffle(all_pairs, labels, random_state=42)

print("Building fused dataset...")

X = []
for img_idx, txt_idx in all_pairs:
    fused = np.concatenate([
        image_features[img_idx],
        text_features[txt_idx]
    ])
    X.append(fused)

X = np.array(X)
y = np.array(labels)

print("Final dataset shape:", X.shape)
print("Label distribution:", np.unique(y, return_counts=True))

np.save("archive/fused_features_smart.npy", X)
np.save("archive/labels_smart.npy", y)

print("Smart balanced dataset created.")