import os
import cv2
import pandas as pd

# ===== PATHS =====
CSV_PATH = r"D:\Final yr Project Dataset\archive\paired_mimic_cxr_with_text.csv"
OUTPUT_DIR = r"D:\Final yr Project Dataset\archive\preprocessed_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224

df = pd.read_csv(CSV_PATH)

print("✅ CSV loaded")
print("Total studies:", len(df))

saved = 0

for idx, row in df.iterrows():
    study_id = row["study_id"]
    image_folder = row["image_folder_path"]

    if not os.path.exists(image_folder):
        continue

    images = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]

    for i, img_name in enumerate(images):
        img_path = os.path.join(image_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        save_name = f"{study_id}_{i}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, (img * 255).astype("uint8"))
        saved += 1

    if idx % 5000 == 0:
        print(f"🖼️ Processed studies: {idx}")

print("🎉 Image preprocessing completed")
print("Total images saved:", saved)
print("Saved at:", OUTPUT_DIR)
