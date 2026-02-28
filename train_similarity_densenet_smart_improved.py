import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

print("🔄 Loading dataset...")
X = np.load("archive/fused_features_smart.npy")
y = np.load("archive/labels_smart.npy")

print("Dataset shape:", X.shape)

# -----------------------
# 🔥 FEATURE NORMALIZATION
# -----------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

import joblib
joblib.dump(scaler, "archive/scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)

class SimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1792, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimilarityModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

print("🚀 Training started...")

for epoch in range(35):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/35 - Loss: {total_loss:.4f}")

print("\n📊 Evaluating...")

model.eval()
with torch.no_grad():
    logits = model(X_test.to(device))
    probs = torch.sigmoid(logits).cpu().numpy()

# Threshold tuning
best_acc = 0
best_thresh = 0.5

for t in np.arange(0.3, 0.7, 0.01):
    preds = (probs > t).astype(int)
    acc = accuracy_score(y_test.numpy(), preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

final_preds = (probs > best_thresh).astype(int)

acc = accuracy_score(y_test.numpy(), final_preds)
prec = precision_score(y_test.numpy(), final_preds)
rec = recall_score(y_test.numpy(), final_preds)

print("\n📊 Evaluation Results (Optimized Threshold)")
print("Best Threshold:", round(best_thresh, 2))
print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))

torch.save(model.state_dict(), "archive/similarity_model_smart_improved.pth")

print("\n✅ Training complete.")