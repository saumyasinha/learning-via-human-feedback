import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from model import DECODER, ENCODER, Resize, VAE_LOSS

from predict import prediction_on_frame

model_name = "vae_may18_vae"
weights = f"vae_weights/{model_name}.h5"

model = load_model(weights, custom_objects={"Resize": Resize, "vae_loss_fn": VAE_LOSS})

ENCODER.load_weights(weights, by_name=True)
DECODER.load_weights(weights, by_name=True)

# z_sample = np.random.normal(loc=0, scale=1, size=(1, 20))
# figure = DECODER.predict(z_sample)[0]
folder = "imgs/0bebd5c1-ea42-4c80-a311-b5f9622accdb"
preds = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename)) / 255
    frame = np.expand_dims(frame, axis=0)
    encoded = ENCODER.predict(frame)
    decoded = DECODER.predict(encoded[2])
    m_decoded = model.predict(frame)
    break  # just do one for now

plt.imshow(np.rint(frame[0] * 255).astype(int))
plt.show()

plt.imshow(np.rint(m_decoded[0] * 255).astype(int))
plt.show()

plt.imshow(np.rint(decoded[0] * 255).astype(int))
plt.show()

