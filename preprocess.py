import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_images(images):
    images = images.astype("float32")
    return preprocess_input(images)

def preprocess_metadata(metadata):
    ages = np.array([m[0] if not np.isnan(m[0]) else 50 for m in metadata])
    ages = ages / 100.0

    sex = np.array([[m[1]] for m in metadata])
    loc = np.array([[m[2]] for m in metadata])

    encoder = OneHotEncoder(sparse=False)
    sex_enc = encoder.fit_transform(sex)
    loc_enc = encoder.fit_transform(loc)

    return np.concatenate([ages.reshape(-1,1), sex_enc, loc_enc], axis=1)