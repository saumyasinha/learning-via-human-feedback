import os

import cv2
from keras.models import load_model

from predict import prediction_on_frame

model_name = "vanilla_cnn"
face_classifier_weights = f"weights/{model_name}.h5"
model = load_model(face_classifier_weights)

folder = "au_data/au_17"
preds = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename))
    if frame is not None:
        preds.append(prediction_on_frame(frame, model, use_cnn=True))


au_names = [
    "Unilateral_RAU14",
    "AU17",
    "AU02",
    "AU25",
    "AU20",
    "Unilateral_RAU12",
    "AU05",
    "Unilateral_LAU14",
    "AU18",
    "AU14",
    "AU26",
    "AU06",
    "Unilateral_LAU12",
    "AU04",
    "Neutral",
]

max_idx_preds = [au_names[p.index(max(p))] for p in preds]
print(max_idx_preds)
