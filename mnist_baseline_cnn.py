"""
Baseline CNN for MNIST handwritten digit classification.

Builds a minimal one-block convolutional network, trains it on MNIST,
saves a block diagram and training curves, and reports test-set metrics.

Run:  python mnist_baseline_cnn.py
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

SEED = 42
IMAGES_DIR = "images"

tf.random.set_seed(SEED)
np.random.seed(SEED)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1), name="Input"),

    layers.Conv2D(32, (3, 3), activation="relu", name="Conv_1"),
    layers.MaxPooling2D((2, 2), name="MaxPool_1"),

    layers.Flatten(name="Flatten"),
    layers.Dense(64, activation="relu", name="Dense_1"),

    layers.Dense(10, activation="softmax", name="Output"),
])

model.summary()


# ---------------------------------------------------------------------------
# 2. Block diagram of the architecture
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")

blocks = [
    ("Input\n28×28×1", "#AED6F1"),
    ("Conv2D\n32 filters\n3×3, ReLU", "#A9DFBF"),
    ("MaxPool\n2×2", "#F9E79F"),
    ("Flatten\n5408", "#F0B27A"),
    ("Dense\n64, ReLU", "#C39BD3"),
    ("Output\n10, Softmax", "#F1948A"),
]

x_start, box_width, box_height, y_center = 0.05, 0.13, 0.55, 0.5

for i, (label, color) in enumerate(blocks):
    x = x_start + i * (box_width + 0.03)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y_center - box_height / 2), box_width, box_height,
        boxstyle="round,pad=0.02", linewidth=1.5, edgecolor="gray", facecolor=color,
    ))
    ax.text(x + box_width / 2, y_center, label,
            ha="center", va="center", fontsize=8.5, fontweight="bold")
    if i < len(blocks) - 1:
        arrow_x = x + box_width + 0.005
        ax.annotate("", xy=(arrow_x + 0.018, y_center), xytext=(arrow_x, y_center),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Baseline CNN — Block Diagram", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "block_diagram.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Data
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale pixels to [0, 1] and add the channel dimension expected by Conv2D
x_train = (x_train / 255.0)[..., np.newaxis]
x_test = (x_test / 255.0)[..., np.newaxis]


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# ---------------------------------------------------------------------------
# 5. Training curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="red", linestyle="--")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="red", linestyle="--")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "training_curves.png"))
plt.close()


# ---------------------------------------------------------------------------
# 6. Test-set evaluation
# ---------------------------------------------------------------------------
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [
        accuracy_score(y_test, y_pred_classes),
        precision_score(y_test, y_pred_classes, average="macro"),
        recall_score(y_test, y_pred_classes, average="macro"),
        f1_score(y_test, y_pred_classes, average="macro"),
        roc_auc_score(y_test, y_pred_probs, multi_class="ovr"),
    ],
})

print("\n==========================================")
print("       TEST SET EVALUATION METRICS        ")
print("==========================================")
print(metrics_df.to_string(index=False))
print("==========================================\n")
