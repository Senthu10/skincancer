import os
import cv2
import pandas as pd
import numpy as np

IMAGE_DIR = "dataset/images"

LABEL_MAP = {
    'nv': 0, 'mel': 1, 'bkl': 2,
    'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
}

def load_data(csv_path, samples_per_class=300):
    df = pd.read_csv(csv_path)

    # Balanced subset
    df = (
        df.groupby('dx')
          .apply(lambda x: x.sample(min(len(x), samples_per_class),
                                    random_state=42))
          .reset_index(drop=True)
    )

    images, labels, metadata = [], [], []

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['image_id'] + ".jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        images.append(img)
        labels.append(LABEL_MAP[row['dx']])
        metadata.append([row['age'], row['sex'], row['localization']])

    return np.array(images), np.array(labels), metadata