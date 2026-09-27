import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# RETINAI - MODEL EVALUATION
# ============================================================

MODEL_PATH = "models/retinaAI_dr_model.keras"

DATASET_FOLDER = "dataset/aptos2019-blindness-detection"
CSV_PATH = os.path.join(DATASET_FOLDER, "train.csv")
IMAGE_FOLDER = os.path.join(DATASET_FOLDER, "train_images")

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

print()
print("=" * 65)
print("RETINAI - MODEL EVALUATION")
print("=" * 65)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("Total labeled images:", len(df))

# ============================================================
# CHECK IMAGE FOLDER
# ============================================================

if not os.path.exists(IMAGE_FOLDER):
    raise FileNotFoundError(
        f"""
Could not find:

{IMAGE_FOLDER}

Your training images must be extracted into:

dataset/train_images/
"""
    )

# ============================================================
# LOAD IMAGES
# ============================================================

images = []
labels = []

missing = 0

print("\nPreparing images...")
print("This may take a few minutes.\n")

for index, row in df.iterrows():

    image_id = str(row["id_code"])
    label = int(row["diagnosis"])

    possible_paths = [
        os.path.join(
            IMAGE_FOLDER,
            image_id + ".png"
        ),
        os.path.join(
            IMAGE_FOLDER,
            image_id + ".jpg"
        ),
        os.path.join(
            IMAGE_FOLDER,
            image_id + ".jpeg"
        )
    ]

    image_path = None

    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        missing += 1
        continue

    try:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224)
        )

        image_array = tf.keras.utils.img_to_array(
            image
        )

        image_array = image_array / 255.0

        images.append(image_array)
        labels.append(label)

    except Exception as e:

        print(
            "Could not process:",
            image_id,
            e
        )

    if (index + 1) % 250 == 0:
        print(
            f"Processed {index + 1}/{len(df)}"
        )

# ============================================================
# CONVERT
# ============================================================

X = np.array(
    images,
    dtype=np.float32
)

y_true = np.array(
    labels,
    dtype=np.int32
)

print()
print("=" * 65)
print("Images successfully loaded:", len(X))
print("Missing images:", missing)
print("=" * 65)

if len(X) == 0:
    raise RuntimeError(
        "No images were loaded."
    )

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning model predictions...")

probabilities = model.predict(
    X,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("RETINAI MODEL PERFORMANCE")
print("=" * 65)

print()
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print()
print("=" * 65)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASS-WISE PERFORMANCE")
print("=" * 65)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nCONFUSION MATRIX")
print("=" * 65)
print(cm)

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

os.makedirs(
    "static/results",
    exist_ok=True
)

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "RetinAI - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=35,
    ha="right"
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

threshold = cm.max() / 2.0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black"
        )

plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")

plt.tight_layout()

confusion_path = (
    "static/results/confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ],

    "Percentage": [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]
})

metrics_path = (
    "static/results/model_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print()
print("Confusion matrix saved:")
print(confusion_path)

print()
print("Metrics saved:")
print(metrics_path)

print()
print("=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)