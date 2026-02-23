import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from src.train import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 7

# Load data
images, labels, metadata = load_data(
    "dataset/balanced_metadata.csv",
    "dataset/images"
)

labels_cat = to_categorical(labels, NUM_CLASSES)

_, X_img_val, _, X_meta_val, _, y_val = train_test_split(
    images, metadata, labels_cat, test_size=0.2, random_state=42
)

# Load model
model = load_model("saved_model/efficientnet_stage2.keras")

# Predict
y_pred = model.predict([X_img_val, X_meta_val])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred_classes))