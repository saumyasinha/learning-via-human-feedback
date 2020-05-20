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

model_name = "may20_vae"
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

# n_batch = 1000
# _, _, z = encoder.predict_generator(valid_gen, n_batch)
# train_z_mean = np.mean(z, axis=0)
# print(train_z_mean)
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

# folder = "imgs/0bebd5c1-ea42-4c80-a311-b5f9622accdb"
folder = "au_data/au_25"
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

classifier = vae_classifier.get_layer("classifier")

orig_preds = []
preds = []
for idx, frame in enumerate(frames):
    encoded = encoder.predict(np.expand_dims(frame, axis=0))
    modified_input = encoded[2] - np.expand_dims(z_mean_diff, axis=0)
    decoded = decoder.predict(modified_input)
    class_pred = classifier.predict(modified_input)
    orig_pred = classifier.predict(encoded[2])
    preds.append(class_pred[0].tolist())
    orig_preds.append(orig_pred[0].tolist())

max_idx_preds = [au_names[p.index(max(p))] for p in preds]
print(f"max_idx_preds: {max_idx_preds}")

orig_max_idx_preds = [au_names[p.index(max(p))] for p in orig_preds]
print(f"original max_idx_preds: {orig_max_idx_preds}")
import ipdb

ipdb.set_trace()
