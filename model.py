import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.models import Model


def build_model(num_classes, metadata_dim):
    # -------- IMAGE BRANCH --------
    image_input = Input(shape=(224, 224, 3), name="image_input")

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=image_input
    )

    # Freeze initially (Stage-1)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    # -------- METADATA BRANCH (REDUCED POWER) --------
    meta_input = Input(shape=(metadata_dim,), name="meta_input")
    m = Dense(16, activation="relu")(meta_input)
    m = Dropout(0.1)(m)

    # -------- FUSION --------
    combined = Concatenate()([x, m])
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.5)(combined)

    output = Dense(num_classes, activation="softmax")(combined)

    model = Model(inputs=[image_input, meta_input], outputs=output)
    return model