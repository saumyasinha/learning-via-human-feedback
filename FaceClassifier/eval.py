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

n_batch = 1000
_, _, z = encoder.predict_generator(valid_gen, n_batch)
train_z_mean = np.mean(z, axis=0)


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
