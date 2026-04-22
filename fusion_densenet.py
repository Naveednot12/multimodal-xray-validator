import numpy as np

print("Loading aligned indices...")
aligned_indices = np.load("archive/aligned_image_indices.npy")

print("Loading DenseNet image features...")
image_features = np.load("archive/image_features_densenet.npy")

print("Loading text features...")
text_features = np.load("archive/text_features.npy")

print("Loading labels...")
labels = np.load("archive/labels.npy")

# Align image features
image_features_aligned = image_features[aligned_indices]

# Since multiple images per study,
# we need to repeat text features accordingly
text_features_aligned = np.repeat(text_features, 2, axis=0)[:len(image_features_aligned)]
labels_aligned = np.repeat(labels, 2)[:len(image_features_aligned)]

print("Image shape:", image_features_aligned.shape)
print("Text shape :", text_features_aligned.shape)

# Fuse
fused = np.concatenate([image_features_aligned, text_features_aligned], axis=1)

print("Fused shape:", fused.shape)

np.save("archive/fused_features_densenet.npy", fused)
np.save("archive/labels_densenet.npy", labels_aligned)

print("Fusion complete.")