import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.model import build_model


IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10   # 🔥 Stage 1
NUM_CLASSES = 7


def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    images = []
    labels = []
    metadata = []

    label_map = {label: idx for idx, label in enumerate(df["dx"].unique())}

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_id"] + ".jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        images.append(img)
        labels.append(label_map[row["dx"]])

        # ---- METADATA ----
        age = row["age"] if not np.isnan(row["age"]) else 0
        sex = 1 if row["sex"] == "male" else 0
        metadata.append([age / 100, sex])

    return np.array(images), np.array(labels), np.array(metadata)


def main():
    print("Loading dataset...")

    images, labels, metadata = load_data(
        "dataset/balanced_metadata.csv",
        "dataset/images"
    )

    labels = to_categorical(labels, NUM_CLASSES)

    X_img_train, X_img_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        images, metadata, labels, test_size=0.2, random_state=42
    )

    model = build_model(NUM_CLASSES, metadata.shape[1])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # 🔥 Lower LR
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("Starting Stage 1 training...")

    model.fit(
        [X_img_train, X_meta_train],
        y_train,
        validation_data=([X_img_val, X_meta_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/efficientnet_stage1.keras")

    print("Stage 1 training complete ✅")


if __name__ == "__main__":
    main()