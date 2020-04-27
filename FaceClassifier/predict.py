import numpy as np
from keras.models import load_model
from FaceClassifier.utils.utils import ImageGenerator
import cv2
import pandas as pd
import argparse

"""
Uses a trained Keras model to make predict the probability distributin of facial action units.
The `prediction` fuction returns a dictionary mapping class names i.e AUs to their probability scores.
See function description for more details about arguments.

"""


def preprocess(img, resize_dims):
    generator = ImageGenerator(to_fit=False)
    img = generator.normalize_img(img)
    img = generator.resize_img(img, resize_dims)
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    return img


def prediction(img_path, model, model_path=None, classes=None):
    """
  Uses a trained Keras model to make predict the probability distributin of facial action units.
  The `prediction` fuction returns a dictionary mapping class names i.e AUs to their probability scores.

  Args:
  - img_path: str, path to image
  - model: keras.models.Model, use a pre-loaded model to increase speed
  - model_path: str, path to the trained model
  - classes: list of classes e.g. ['AU04','AU05']

  Returns:
  A dictionary mapping the class labels to the precicted probability scores

  """

    # read the image and preprocess it
    img = cv2.imread(img_path)
    img = preprocess(img, resize_dims=(model.input_shape[1], model.input_shape[2]))
    predictions = model.predict(img).flatten()
    if classes is None:
        classes = list(np.arange(0, len(list(predictions))))

    return dict(zip(classes, list(predictions)))


def prediction_on_frame(frame, model, model_path=None, classes=None):
    """
  Uses a trained Keras model to make predict the probability distributin of facial action units.
  The `prediction` fuction returns a dictionary mapping class names i.e AUs to their probability scores.

  Args:
  - img_path: str, path to image
  - model: keras.models.Model, use a pre-loaded model
  - classes: list of classes e.g. ['AU04','AU05']

  Returns:
  A dictionary mapping the class labels to the precicted probability scores

  """

    # read the image and preprocess it
    img = preprocess(frame, resize_dims=(model.input_shape[1], model.input_shape[2]))
    predictions = list(model.predict(img).flatten())
    if classes is None:
        classes = list(np.arange(0, len(predictions)))

    return dict(zip(classes, predictions))
