import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# RETINAI - DIABETIC RETINOPATHY TRAINING
# ============================================================

print("\n==========================================")
print("RETINAI MODEL TRAINING")
print("==========================================\n")


# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "aptos2019-blindness-detection",
    "train.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "aptos2019-blindness-detection",
    "train_images"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# SETTINGS
# -----------------------------

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12
NUM_CLASSES = 5
SEED = 42


# -----------------------------
# LOAD CSV
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")

print("\nClass distribution:")
print(df["diagnosis"].value_counts().sort_index())


# -----------------------------
# IMAGE PATHS
# -----------------------------

def get_image_path(image_id):

    extensions = [
        ".png",
        ".jpg",
        ".jpeg"
    ]

    for ext in extensions:

        path = os.path.join(
            IMAGE_DIR,
            image_id + ext
        )

        if os.path.exists(path):
            return path

    return None


df["image_path"] = df["id_code"].apply(get_image_path)

# Remove missing images
df = df.dropna(subset=["image_path"]).reset_index(drop=True)

print(f"\nImages found: {len(df)}")


# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["diagnosis"]
)

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")


# -----------------------------
# DATA AUGMENTATION
# -----------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")


# -----------------------------
# IMAGE LOADING FUNCTION
# -----------------------------

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# -----------------------------
# CREATE DATASETS
# -----------------------------

train_paths = train_df["image_path"].values
train_labels = train_df["diagnosis"].values

val_paths = val_df["image_path"].values
val_labels = val_df["diagnosis"].values


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(
    1000,
    seed=SEED
)

train_dataset = train_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(BATCH_SIZE)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# -----------------------------
# CLASS WEIGHTS
# -----------------------------

classes = np.unique(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)


# -----------------------------
# BASE MODEL
# -----------------------------

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# -----------------------------
# MODEL
# -----------------------------

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# -----------------------------
# COMPILE
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# MODEL SUMMARY
# -----------------------------

model.summary()


# -----------------------------
# CALLBACKS
# -----------------------------

model_path = os.path.join(
    MODEL_DIR,
    "retinaAI_dr_model.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# -----------------------------
# TRAIN
# -----------------------------

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================\n")

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# -----------------------------
# SAVE MODEL
# -----------------------------

model.save(model_path)


print("\n==========================================")
print("TRAINING COMPLETE!")
print("==========================================")

print("\nModel saved at:")
print(model_path)

print("\nRetinAI model is ready.")

print("\nIMPORTANT:")
print("This is an AI-assisted screening prototype.")
print("It is NOT a medical diagnosis.")