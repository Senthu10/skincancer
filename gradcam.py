import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "saved_model/skin_cancer_efficientnetb0_stage2.h5"
CSV_PATH   = "dataset/balanced_metadata.csv"
IMAGE_PATH = "dataset/images/ISIC_0030095.jpg"
IMG_SIZE   = 224
LAST_CONV_LAYER = "top_conv"   # EfficientNetB0

# -----------------------------
# CLASS MAPPING
# -----------------------------
CLASS_NAMES = {
    0: "Melanocytic Nevi (NV)",
    1: "Melanoma (MEL)",
    2: "Benign Keratosis-like Lesions (BKL)",
    3: "Basal Cell Carcinoma (BCC)",
    4: "Actinic Keratoses (AKIEC)",
    5: "Dermatofibroma (DF)",
    6: "Vascular Lesions (VASC)"
}

CLASS_CODE = {
    0: "nv",
    1: "mel",
    2: "bkl",
    3: "bcc",
    4: "akiec",
    5: "df",
    6: "vasc"
}

# Malignant classes
MALIGNANT_CLASSES = [1, 3, 4]  # MEL, BCC, AKIEC

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# -----------------------------
# LOAD & PREPROCESS IMAGE
# -----------------------------
img = load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# LOAD METADATA + GROUND TRUTH (SAFE)
# -----------------------------
df = pd.read_csv(CSV_PATH)

image_id = os.path.basename(IMAGE_PATH).replace(".jpg", "")
row_match = df[df["image_id"] == image_id]

true_dx = None

if len(row_match) == 0:
    print("⚠️ Metadata not found for image. Using default values.")
    age = 0.0
    sex = 0.0
else:
    row = row_match.iloc[0]

    age = row["age"]
    age = 0 if pd.isna(age) else age / 100.0

    sex = row["sex"]
    sex = 1 if sex == "male" else 0

    true_dx = row["dx"]

meta_features = np.array([[age, sex]], dtype="float32")
print("✅ Metadata used:", meta_features)

# -----------------------------
# GRAD-CAM FUNCTION (MULTI-INPUT)
# -----------------------------
def make_gradcam_heatmap(inputs, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), class_index.numpy()

# -----------------------------
# GENERATE GRAD-CAM
# -----------------------------
heatmap, pred_class = make_gradcam_heatmap(
    [img_array, meta_features],
    model,
    LAST_CONV_LAYER
)

# -----------------------------
# PREDICTION INTERPRETATION
# -----------------------------
lesion_type = CLASS_NAMES[pred_class]
predicted_dx = CLASS_CODE[pred_class]

binary_type = "Malignant" if pred_class in MALIGNANT_CLASSES else "Benign"

print("\n==============================")
print("Binary Type   :", binary_type)
print("Lesion Found  :", lesion_type)
print("==============================")

# -----------------------------
# CORRECTNESS CHECK
# -----------------------------
if true_dx is None:
    print("❌ Ground truth not available for this image")
elif predicted_dx == true_dx:
    print("✅ Prediction MATCHES dataset label")
else:
    print("❌ Prediction DOES NOT match dataset label")

print("Ground Truth dx :", true_dx)
print("Predicted dx    :", predicted_dx)
print("==============================\n")

# -----------------------------
# OVERLAY HEATMAP
# -----------------------------
orig = cv2.imread(IMAGE_PATH)
orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))

heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

# -----------------------------
# DISPLAY RESULT
# -----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title(f"{binary_type}\n{lesion_type}")
plt.axis("off")
plt.show()