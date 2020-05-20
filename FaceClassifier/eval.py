import os

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split

from model import VAE_LOSS
from predict import prediction_on_frame
from utils.utils import ImageGenerator, Resize



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
vae_classifier.get_layer("")

# find mean z over a training samples
df = pd.read_csv("data/master.csv")
num_classes = len(df.columns[1:])

# split data into training set and validation set
X_train, X_val = train_test_split(
    list(df.iloc[:, 0]), shuffle=True, test_size=0.2, random_state=42
)

batch_size = 16
valid_gen = ImageGenerator(
    df=df,
    image_dir="imgs",
    image_list=X_val,
    num_classes=num_classes,
    input_shape=(240, 320),
    batch_size=batch_size,
    num_channels=3,
    augmentation=None,
    augment=False,
    image_format="png",
    shuffle=True,
)

n_batch = 10
_, _, z = encoder.predict_generator(valid_gen, n_batch)
train_z_mean = np.mean(z, axis=0)


# z_sample = np.random.normal(loc=0, scale=1, size=(1, 50))
# frame = decoder.predict(z_sample)
# plt.imshow(np.rint(frame[0] * 255).astype(int))
# plt.show()

folder = "imgs/0bebd5c1-ea42-4c80-a311-b5f9622accdb"
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

z_mean_diff = train_z_mean - domain_z_mean

for idx, frame in enumerate(frames):
    encoded = encoder.predict(np.expand_dims(frame, axis=0))
    modified_input = encoded[2] - np.expand_dims(z_mean_diff, axis=0)
    decoded = decoder.predict(modified_input)
    preds = vae_classifier.predict(modified_input)
    max_idx_preds = [au_names[p.index(max(p))] for p in preds]
    print(max_idx_preds)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text("Original")
    ax1.imshow(np.rint(frame[0] * 255).astype(int))

    # ax2.title.set_text("VAE e2e")
    # ax2.imshow(np.rint(m_decoded[0] * 255).astype(int))

    ax3.title.set_text("VAE separate")
    ax3.imshow(np.rint(decoded[0] * 255).astype(int))

    # ax4.title.set_text("VAE modified encoder")
    # ax4.imshow(np.rint(c_decoded[0] * 255).astype(int))

    plt.show()
    break
