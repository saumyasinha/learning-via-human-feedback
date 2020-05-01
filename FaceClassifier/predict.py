import numpy as np
from FaceClassifier.utils.utils import ImageGenerator, DataGenerator, get_landmark_points
import cv2
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

def prediction_on_frame(frame, model, use_cnn, detector=None, predictor=None, classes=None):
    """
    Uses a trained model to predict the probability distribution of facial action units.
    The fuction returns a dictionary mapping class names i.e AUs to their probability scores.

    Args:
    - frame: numpy array, image
    - model: keras.models.Model, use a pre-loaded model
    - classes: list of classes e.g. ['AU04','AU05']
    - detector: dlib facial landmarks detector
    - predictor: dlib facial landmarks predictor
    - use_cnn: bool, set to True to use CNN model

    Returns:
      A dictionary mapping the class labels to the precicted probability scores

    """
    # if use_cnn, use the CNN model's preprocessing
    if use_cnn:
        # read the image and preprocess it
        img = preprocess(frame, resize_dims=(model.input_shape[1], model.input_shape[2]))
        prediction = list(model.predict(img).flatten())
    else: # use the landmarks model
        img = cv2.resize(frame,(320,240))
        landmarks = get_landmark_points(img, detector, predictor).flatten()
        prediction = list(model.predict(np.reshape(landmarks, (1, 136))).flatten())

    if classes is None:
        classes = list(np.arange(0, len(prediction)))

    return dict(zip(classes, prediction))
