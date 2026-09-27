import os
import uuid
import numpy as np
import tensorflow as tf
import cv2

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image


# ============================================================
# RETINAI
# Explainable AI for Diabetic Retinopathy Screening
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "models/retinaAI_dr_model.keras"


# ============================================================
# DR CLASSES
# ============================================================

CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}


CLASS_EXPLANATIONS = {
    0: "No visible signs of diabetic retinopathy were detected by the AI screening model.",

    1: "The screening model detected features that may be consistent with mild diabetic retinopathy.",

    2: "The screening model detected features that may be consistent with moderate diabetic retinopathy.",

    3: "The screening model detected features that may be consistent with severe diabetic retinopathy.",

    4: "The screening model detected features that may be consistent with proliferative diabetic retinopathy."
}


CLASS_ACTIONS = {
    0: "Continue routine eye screening as recommended by an eye-care professional.",

    1: "An eye-care professional should review the result and determine appropriate follow-up.",

    2: "Further evaluation by an eye-care professional is recommended.",

    3: "Prompt evaluation by an eye-care professional is recommended.",

    4: "Urgent evaluation by an eye-care professional is recommended."
}


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("RETINAI - LOADING MODEL")
print("=" * 60)

model = None

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", MODEL_PATH)
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)

    print()
    print("MODEL LAYERS:")

    for i, layer in enumerate(model.layers):

        print(
            i,
            "|",
            layer.name,
            "|",
            layer.__class__.__name__
        )

except Exception as e:

    print("MODEL LOAD ERROR:", e)

    model = None


# ============================================================
# FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(filepath):

    image = Image.open(
        filepath
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# FIND MOBILENETV2
# ============================================================

def find_base_model():

    if model is None:
        return None

    # Known layer from your architecture
    try:

        return model.get_layer(
            "mobilenetv2_1.00_224"
        )

    except Exception:

        pass


    # Fallback search
    for layer in model.layers:

        if (
            "mobilenet" in layer.name.lower()
            and
            isinstance(
                layer,
                tf.keras.Model
            )
        ):

            return layer


    return None


# ============================================================
# REAL GRAD-CAM
# ============================================================

def generate_gradcam(
    image_array,
    predicted_class,
    original_path
):

    print()
    print("=" * 60)
    print("GENERATING REAL GRAD-CAM")
    print("=" * 60)

    try:

        if model is None:

            print("Model unavailable.")
            return None


        # ----------------------------------------------------
        # FIND MOBILE NET
        # ----------------------------------------------------

        base_model = find_base_model()


        if base_model is None:

            print(
                "ERROR: MobileNetV2 base model not found."
            )

            return None


        print(
            "Base model:",
            base_model.name
        )


        # ----------------------------------------------------
        # FIND OUT_RELU
        # ----------------------------------------------------

        try:

            target_layer = base_model.get_layer(
                "out_relu"
            )

        except Exception:

            target_layer = None


        # Fallback search for 4D feature layer
        if target_layer is None:

            for layer in reversed(
                base_model.layers
            ):

                try:

                    shape = layer.output.shape

                    if len(shape) == 4:

                        target_layer = layer
                        break

                except Exception:

                    continue


        if target_layer is None:

            print(
                "ERROR: Could not find target feature layer."
            )

            return None


        print(
            "Target layer:",
            target_layer.name
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We build the gradient model INSIDE MobileNetV2.
        # Then we manually run the classifier layers after
        # MobileNetV2.
        # ----------------------------------------------------

        internal_model = tf.keras.models.Model(

            inputs=base_model.input,

            outputs=[
                target_layer.output,
                base_model.output
            ]

        )


        # ----------------------------------------------------
        # FIND CLASSIFIER HEAD
        # ----------------------------------------------------

        base_index = model.layers.index(
            base_model
        )


        head_layers = model.layers[
            base_index + 1:
        ]


        print()
        print("Classifier head:")


        for layer in head_layers:

            print(
                " ->",
                layer.name
            )


        # ----------------------------------------------------
        # GRADIENT CALCULATION
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_output, base_output = internal_model(
                image_array,
                training=False
            )


            x = base_output


            # Pass MobileNet output through
            # the remaining classifier layers.
            for layer in head_layers:

                # Skip InputLayer if present
                if isinstance(
                    layer,
                    tf.keras.layers.InputLayer
                ):

                    continue


                x = layer(
                    x,
                    training=False
                )


            predictions = x


            class_score = predictions[
                :,
                predicted_class
            ]


        # ----------------------------------------------------
        # GRADIENTS
        # ----------------------------------------------------

        gradients = tape.gradient(
            class_score,
            conv_output
        )


        if gradients is None:

            print(
                "ERROR: Gradients are None."
            )

            return None


        print(
            "Gradients calculated successfully."
        )


        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )


        conv_output = conv_output[0]

        pooled_gradients = pooled_gradients[0]


        # ----------------------------------------------------
        # WEIGHT FEATURE MAPS
        # ----------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_output *
            pooled_gradients,
            axis=-1
        )


        # ----------------------------------------------------
        # RELU
        # ----------------------------------------------------

        heatmap = tf.maximum(
            heatmap,
            0
        )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        max_value = tf.reduce_max(
            heatmap
        )


        if float(max_value) > 0:

            heatmap = (
                heatmap /
                max_value
            )


        heatmap = heatmap.numpy()


        # ----------------------------------------------------
        # CONVERT TO IMAGE
        # ----------------------------------------------------

        heatmap = np.uint8(
            heatmap * 255
        )


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # SMOOTH
        # ----------------------------------------------------

        heatmap = cv2.GaussianBlur(
            heatmap,
            (0, 0),
            3
        )


        # ----------------------------------------------------
        # COLOUR MAP
        # ----------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

        original = cv2.imread(
            original_path
        )


        if original is None:

            print(
                "ERROR: Could not load original image."
            )

            return None


        original = cv2.resize(
            original,
            (224, 224)
        )


        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap_color,
            0.40,
            0
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        cv2.rectangle(
            overlay,
            (0, 0),
            (224, 28),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            overlay,
            "RetinAI Grad-CAM",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # UNIQUE FILE
        # ----------------------------------------------------

        unique_id = uuid.uuid4().hex[:12]

        filename = (
            "gradcam_" +
            unique_id +
            ".jpg"
        )


        output_path = os.path.join(
            RESULT_FOLDER,
            filename
        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        saved = cv2.imwrite(
            output_path,
            overlay
        )


        if not saved:

            print(
                "ERROR: OpenCV could not save Grad-CAM."
            )

            return None


        print()
        print(
            "SUCCESS!"
        )

        print(
            "Grad-CAM file:",
            output_path
        )


        browser_url = (
            "/" +
            output_path.replace(
                "\\",
                "/"
            )
        )


        print(
            "Browser URL:",
            browser_url
        )


        return browser_url


    except Exception as e:

        print()
        print("=" * 60)
        print("GRAD-CAM ERROR")
        print("=" * 60)

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print("=" * 60)

        return None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if model is None:

        return jsonify({
            "error":
            "AI model could not be loaded."
        }), 500


    if "image" not in request.files:

        return jsonify({
            "error":
            "No retinal image uploaded."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "error":
            "No image selected."
        }), 400


    if not allowed_file(
        file.filename
    ):

        return jsonify({
            "error":
            "Please upload JPG, JPEG or PNG image."
        }), 400


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_name = secure_filename(
        file.filename
    )


    unique_id = uuid.uuid4().hex[:10]


    filename = (
        unique_id +
        "_" +
        safe_name
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    file.save(filepath)


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    try:

        image_array = preprocess_image(
            filepath
        )

    except Exception as e:

        return jsonify({
            "error":
            "Could not process image: " +
            str(e)
        }), 400


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_class = int(
            np.argmax(
                predictions
            )
        )


        confidence = float(
            predictions[
                predicted_class
            ] * 100
        )


        print()
        print("=" * 60)
        print("PREDICTION")
        print("=" * 60)

        print(
            "Class:",
            predicted_class
        )

        print(
            "Prediction:",
            CLASS_NAMES[
                predicted_class
            ]
        )

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )


        print()
        print("ALL CLASS PROBABILITIES:")

        for i in range(5):

            print(
                f"{i}: "
                f"{CLASS_NAMES[i]} = "
                f"{predictions[i] * 100:.2f}%"
            )


    except Exception as e:

        return jsonify({
            "error":
            "Prediction failed: " +
            str(e)
        }), 500


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    heatmap_path = generate_gradcam(
        image_array,
        predicted_class,
        filepath
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    result = {

        "prediction":
        CLASS_NAMES[
            predicted_class
        ],

        "grade":
        predicted_class,

        "confidence":
        f"{confidence:.2f}%",

        "explanation":
        CLASS_EXPLANATIONS[
            predicted_class
        ],

        "recommendation":
        CLASS_ACTIONS[
            predicted_class
        ],

        "clinical_note":
        "This is an AI-assisted screening result "
        "and not a medical diagnosis. Clinical "
        "confirmation by a qualified eye-care "
        "professional is required.",

        "image":
        "/" +
        filepath.replace(
            "\\",
            "/"
        ),

        "heatmap":
        heatmap_path

    }


    print()
    print(
        "Heatmap response:",
        heatmap_path
    )


    return jsonify(
        result
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
        "online",

        "model_loaded":
        model is not None,

        "system":
        "RetinAI"

    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("RETINAI SCREENING SYSTEM")
    print("=" * 60)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(
        debug=True
    )