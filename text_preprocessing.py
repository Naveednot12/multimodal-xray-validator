import pandas as pd
import os
import re

# ================= PATHS =================
INPUT_CSV = r"D:\Final yr Project Dataset\archive\paired_mimic_cxr.csv"
OUTPUT_CSV = r"D:\Final yr Project Dataset\archive\paired_mimic_cxr_with_text.csv"

print("🔹 Script started")

# ================= CHECK INPUT CSV =================
if not os.path.exists(INPUT_CSV):
    print("INPUT CSV NOT FOUND")
    exit()

print("Input CSV found")

df = pd.read_csv(INPUT_CSV)
print(f"Rows in CSV: {len(df)}")

# ================= FUNCTIONS =================
def extract_section(text):
    text = text.lower()

    imp = re.search(r"impression:(.*)", text, re.DOTALL)
    if imp:
        return imp.group(1)

    find = re.search(r"findings:(.*)", text, re.DOTALL)
    if find:
        return find.group(1)

    return text


def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================= PROCESS REPORTS =================
cleaned_reports = []

for i, row in df.iterrows():
    report_path = row["report_path"]

    if not os.path.exists(report_path):
        cleaned_reports.append("")
        continue

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    section = extract_section(raw)
    cleaned = clean_text(section)
    cleaned_reports.append(cleaned)

    if i % 5000 == 0:
        print(f"🔄 Processed {i} reports")

print("Report processing finished")

# ================= SAVE OUTPUT =================
df["cleaned_report"] = cleaned_reports
df.to_csv(OUTPUT_CSV, index=False)

print("OUTPUT FILE CREATED")
print(f"Saved at: {OUTPUT_CSV}")
