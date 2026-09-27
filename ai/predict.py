import os
import numpy as np
import tensorflow as tf
import cv2

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "retina_efficientnet.keras"
)

IMG_SIZE = 224

CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original = image.copy()

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image, original


def predict_image(image_path):

    model = load_model()

    image, original = preprocess_image(
        image_path
    )

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_class]
    )

    return {
        "class_id": predicted_class,
        "diagnosis": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": predictions.tolist()
    }


def generate_gradcam(
    image_path,
    output_path
):

    model = load_model()

    image, original = preprocess_image(
        image_path
    )

    # EfficientNet feature-extraction layer
    base_model = model.layers[0]

    last_conv_layer = None

    for layer in reversed(base_model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError(
            "Could not find convolutional layer."
        )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:
        heatmap /= max_value

    heatmap = heatmap.numpy()

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    cv2.imwrite(
        output_path,
        overlay
    )

    return output_path


if __name__ == "__main__":

    print("Prediction module ready.")

    print("\nExpected model:")
    print(MODEL_PATH)

    print("\nDR classes:")

    for key, value in CLASS_NAMES.items():
        print(f"{key}: {value}")