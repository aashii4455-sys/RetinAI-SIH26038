import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ============================================================
# SIH26038 - Diabetic Retinopathy Training Pipeline
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATASET_DIR, "train_images")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 5
SEED = 42


print("=" * 60)
print("SIH26038 - DIABETIC RETINOPATHY AI TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# Check dataset
# ------------------------------------------------------------

if not os.path.exists(CSV_PATH):
    print("\nERROR: train.csv was not found.")
    print("Expected location:")
    print(CSV_PATH)
    print("\nDownload the APTOS training data first.")
    raise SystemExit

if not os.path.exists(IMAGE_DIR):
    print("\nERROR: train_images folder was not found.")
    print("Expected location:")
    print(IMAGE_DIR)
    print("\nExtract the APTOS training images into dataset/train_images.")
    raise SystemExit


# ------------------------------------------------------------
# Read labels
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print(f"\nLabels found: {len(df)}")

print("\nClass distribution:")
print(df["diagnosis"].value_counts().sort_index())


# ------------------------------------------------------------
# Locate image files
# ------------------------------------------------------------

image_files = {}

extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

for extension in extensions:
    for path in glob.glob(os.path.join(IMAGE_DIR, extension)):
        filename = os.path.splitext(os.path.basename(path))[0]
        image_files[filename] = path


print(f"\nImages found: {len(image_files)}")


# ------------------------------------------------------------
# Match CSV labels with images
# ------------------------------------------------------------

df["filepath"] = df["id_code"].map(image_files)

missing = df["filepath"].isna().sum()

if missing > 0:
    print(f"\nWARNING: {missing} images could not be matched.")
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)

print(f"Matched samples: {len(df)}")


if len(df) == 0:
    print("\nERROR: No images matched train.csv.")
    raise SystemExit


# ------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["diagnosis"]
)

print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)


# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------

classes = np.unique(train_df["diagnosis"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["diagnosis"]
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")
print(class_weights)


# ------------------------------------------------------------
# EfficientNetB0 model
# ------------------------------------------------------------

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False


model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.35),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.25),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "retina_efficientnet.keras"
)

callbacks = [

    ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

final_model_path = os.path.join(
    MODEL_DIR,
    "retina_efficientnet_final.keras"
)

model.save(final_model_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nBest model:")
print(model_path)

print("\nFinal model:")
print(final_model_path)

print("\nDR Classes:")
print("0 = No Diabetic Retinopathy")
print("1 = Mild")
print("2 = Moderate")
print("3 = Severe")
print("4 = Proliferative Diabetic Retinopathy")