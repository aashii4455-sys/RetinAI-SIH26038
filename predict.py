import os
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# RETINAAI PREDICTION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "retinaaI_dr_model.keras"
)

IMG_SIZE = 224

CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}


def predict_image(image_path):

    print("\nLoading AI model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Prediction
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class]) * 100

    result = CLASS_NAMES[predicted_class]

    print("\n==============================")
    print("RETINAAI SCREENING RESULT")
    print("==============================")

    print("Prediction:", result)
    print("DR Grade:", predicted_class)
    print("Confidence: {:.2f}%".format(confidence))

    print("\nAll class probabilities:")

    for i, probability in enumerate(predictions):
        print(
            "{}: {:.2f}%".format(
                CLASS_NAMES[i],
                probability * 100
            )
        )

    print("==============================")

    return predicted_class, result, confidence


# Test from command line
if __name__ == "__main__":

    image_path = input(
        "\nEnter the full path of a retinal image: "
    ).strip('"')

    if not os.path.exists(image_path):
        print("\nERROR: Image not found.")
    else:
        predict_image(image_path)