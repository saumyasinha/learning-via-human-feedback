import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from utils.utils import Resize
from model import VAE_LOSS

from predict import prediction_on_frame


train_z_mean = np.array(
    [
        0.02873915,
        -0.03757364,
        -0.18720372,
        0.02146574,
        0.21144809,
        -0.06107251,
        -0.02077978,
        -0.09831361,
        -0.00233703,
        0.02472817,
        0.01851729,
        -0.08540793,
        -0.00896718,
        0.11292063,
        -0.10131415,
        0.12061853,
        -0.00947284,
        -0.9389922,
        -0.01437521,
        0.10037095,
        0.17108364,
        -0.02416838,
        -0.02745124,
        -0.06421229,
        0.01934998,
        0.00115275,
        0.02176926,
        0.09721659,
        0.23399407,
        -0.02147517,
        0.01799874,
        0.05654962,
        0.0513767,
        -0.09091707,
        0.0923245,
        -0.02486692,
        0.11765029,
        -0.04970305,
        -0.05292111,
        0.16274576,
        -0.51810914,
        0.13472795,
        0.01017204,
        0.0192815,
        0.07664829,
        -0.12937498,
        -0.03471367,
        0.08723726,
        0.03432051,
        -0.13011713,
    ]
)
model_name = "may19_vae"
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
# for ind, w in enumerate(c_encoder_weights):
#     assert (encoder_weights[ind] == w).all()


# z_sample = np.random.normal(loc=0, scale=1, size=(1, 50))
# frame = decoder.predict(z_sample)
# plt.imshow(np.rint(frame[0] * 255).astype(int))
# plt.show()

# folder = "imgs/0bebd5c1-ea42-4c80-a311-b5f9622accdb"
folder = "au_data/au_12"
frames = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename))
    if frame is not None:
        frame = frame / 255
        frames.append(frame)
frames = np.array(frames)
_, _, z = encoder.predict(frames)
# Just take the mean of all frames (could use first n frames or rolling mean)
domain_z_mean = np.mean(z, axis=0)
z_mean_diff = domain_z_mean - train_z_mean

preds = []
for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder, filename))
    if frame is not None:
        frame = frame / 255
        frame = np.expand_dims(frame, axis=0)

        encoded = encoder.predict(frame)
        modified_input = encoded[2] - np.expand_dims(z_mean_diff, axis=0)
        mod_decoded = decoder.predict(modified_input)

        decoded = decoder.predict(encoded[2])
        c_encoded = c_encoder.predict(frame)
        c_decoded = decoder.predict(c_encoded[2])
        m_decoded = vae_model.predict(frame)

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.title.set_text("Original")
        ax1.imshow(np.rint(frame[0] * 255).astype(int))

        ax2.title.set_text("VAE e2e")
        ax2.imshow(np.rint(m_decoded[0] * 255).astype(int))

        ax3.title.set_text("VAE separate")
        ax3.imshow(np.rint(decoded[0] * 255).astype(int))

        ax4.title.set_text("VAE modified encoder")
        ax4.imshow(np.rint(mod_decoded[0] * 255).astype(int))

        plt.show()
        break
