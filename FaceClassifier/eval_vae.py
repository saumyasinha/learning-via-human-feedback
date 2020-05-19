import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from utils.utils import Resize
from model import VAE_LOSS

from predict import prediction_on_frame

model_name = "may18_vae"
vae_model = load_model(
    f"vae_weights/vae_{model_name}.h5",
    custom_objects={"Resize": Resize, "vae_loss_fn": VAE_LOSS},
)

vae_classifier = load_model(
    f"vae_weights/{model_name}.h5",
    custom_objects={"Resize": Resize, "vae_loss_fn": VAE_LOSS},
)

decoder = vae_model.get_layer("decoder")
encoder = vae_model.get_layer("encoder")
encoder_weights = encoder.get_weights()
c_encoder = vae_classifier.get_layer("encoder")
c_encoder_weights = c_encoder.get_weights()
# TODO: encoder weights not frozen
# for ind, w in enumerate(c_encoder_weights):
#     assert (encoder_weights[ind] == w).all()


# z_sample = np.random.normal(loc=0, scale=1, size=(1, 20))
# figure = DECODER.predict(z_sample)[0]
folder = "imgs/0bebd5c1-ea42-4c80-a311-b5f9622accdb"
preds = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename))
    if frame is not None:
        frame = frame / 255
        frame = np.expand_dims(frame, axis=0)
        encoded = encoder.predict(frame)
        decoded = decoder.predict(encoded[2])
        m_decoded = vae_model.predict(frame)
        break  # just do one for now

plt.imshow(np.rint(frame[0] * 255).astype(int))
plt.show()

plt.imshow(np.rint(m_decoded[0] * 255).astype(int))
plt.show()

plt.imshow(np.rint(decoded[0] * 255).astype(int))
plt.show()
