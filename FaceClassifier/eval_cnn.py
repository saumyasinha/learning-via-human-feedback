import os

import cv2
from keras.models import load_model

from predict import prediction_on_frame

model_name = "vanilla_cnn"
face_classifier_weights = f"weights/{model_name}.h5"
model = load_model(face_classifier_weights)

folder = "au_data/au_12"
preds = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename))
    if frame is not None:
        preds.append(prediction_on_frame(frame, model, use_cnn=True))

max_idx_preds = [ p.index(min(p)) for p in preds]
print(max_idx_preds)