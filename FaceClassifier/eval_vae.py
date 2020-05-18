import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from model import DECODER, ENCODER

from predict import prediction_on_frame

model_name = "vae_overnight"
weights = f"weights/{model_name}.h5"

ENCODER.load_weights(weights, by_name=True)
DECODER.load_weights(weights, by_name=True)
DECODER.summary()

height = 240
width = 320

# z_sample = np.random.normal(loc=0, scale=1, size=(1, 20))
# figure = DECODER.predict(z_sample)[0]
folder = "imgs/0bebd5c1-ea42-4c80-a311-b5f9622accdb"
preds = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename))
    encoded = ENCODER.predict(np.expand_dims(frame, axis=0))
    decoded = DECODER.predict(encoded[2])
    break  # just do one for now

plt.imshow(frame)
plt.show()
plt.imshow(decoded[0])
plt.show()
