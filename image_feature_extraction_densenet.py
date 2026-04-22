import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchxrayvision as xrv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load medical DenseNet
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model = model.features
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

BASE_PATH = r"D:\Final yr Project Dataset\archive\mimic-cxr-dataset\official_data_iccv_final\files"

image_paths = []

print("Indexing images...")

for root, dirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))

print("Total images found:", len(image_paths))

features = []

print("Extracting DenseNet features...")

for img_path in tqdm(image_paths):

    image = Image.open(img_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(image)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(1, -1)

    features.append(feat.cpu().numpy())

features = np.vstack(features)

print("Feature shape:", features.shape)

np.save("archive/image_features_densenet.npy", features)
np.save("archive/image_paths_densenet.npy", np.array(image_paths))

print("DenseNet feature extraction complete.")