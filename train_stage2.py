import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from src.train import load_data


BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 7


def main():
    print("Loading data...")
    images, labels, metadata = load_data(
        "dataset/balanced_metadata.csv",
        "dataset/images"
    )

    labels_cat = to_categorical(labels, NUM_CLASSES)

    X_img_train, X_img_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
        images, metadata, labels_cat, test_size=0.2, random_state=42
    )

    print("Loading Stage-1 model...")
    model = load_model("saved_model/efficientnet_stage1.keras")

    # 🔓 UNFREEZE TOP EfficientNet BLOCKS
    for layer in model.layers:
        if layer.name.startswith(("block4", "block5", "block6", "block7")):
            layer.trainable = True
        else:
            layer.trainable = False

    # 🎯 CLASS WEIGHTS (CRITICAL FIX)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.argmax(y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    model.compile(
       optimizer=Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("Starting Stage-2 fine-tuning...")
    model.fit(
        [X_img_train, X_meta_train],
        y_train,
        validation_data=([X_img_val, X_meta_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=4, restore_best_weights=True),
            ReduceLROnPlateau(patience=2)
        ]
    )

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/skin_cancer_efficientnetb0_stage2.h5")

    print("Stage-2 complete ✅")


if __name__ == "__main__":
    main()