import os

import cv2
import dlib
import keras.backend as K
import numpy as np
import tensorflow as tf
from imutils import face_utils
from keras.callbacks import Callback
from keras.layers import BatchNormalization, Layer
from keras.utils import Sequence, to_categorical


def BatchNorm():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)


class Resize(Layer):
    """ Custom Keras layer that resizes to a new size using interpolation.
    Bypasses the use of Keras Lambda layer
    Args:
      - new_size: tuple, new size to which layer needs to be resized to. Must be (height, width)
      - method: str, method of interpolation to be used. If None, defaults to bilinear.
               Choose amongst 'bilinear', 'nearest', 'lanczos3', 'lanczos5', 'area', 'gaussian', 'mitchellcubic'
    Returns:
      - keras.layers.Layer of size [?, new_size[0], new_size[1], depth]
    """

    def __init__(self, new_size, method="bilinear", **kwargs):
        self.new_size = new_size
        self.method = method
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Resize, self).build(input_shape)

    def call(self, inputs, **kwargs):
        resized_height, resized_width = self.new_size
        return tf.image.resize(
            images=inputs,
            size=[resized_height, resized_width],
            method=self.method,
            # align_corners=True,
        )

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Resize, self).get_config()
        config["new_size"] = self.new_size
        return config


class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight

    def on_epoch_end(self, epoch, logs={}):
        if epoch > klstart:
            new_weight = min(K.get_value(self.weight) + (1.0 / annealtime), 1.0)
            K.set_value(self.weight, new_weight)
        print("Current KL Weight is " + str(K.get_value(self.weight)))


def get_landmark_points(img, detector, predictor):
    """
    Function to detect facial landmarks in an image. Returns a numpy array containing
    the x and y coordinates of the landmarks.
    """
    # Converting the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    if len(rects) > 0:
        rect = rects[0]
        # Find the landmark points
        # Make the prediction and transform it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape
    else:
        return np.empty(0)


class LandmarkDataGenerator(Sequence):
    """
  Image generator inherited from keras.utils.Sequence to be used for dense network (landmarks).
  Args:
  - ground_truth: dict, dictionary containing the filenames as keys, ground truth AUs as a flattned numpy array
  - input_data: dict, dictionary containing the filenames as keys, input data (landmarks) as a flattned numpy array
  - image_list: list of images filenames
  - num_classes: int, number of classes to classify images into
  - batch_size: int, size of batch to be used during training
  - input_shape: int, specifying shape that matches with the model's input shape
  - shuffle: bool, set to True to shuffle the dataset
  - to_fit: bool, set to True during training, set to False during prediction.

  Returns:
  - X: images to be used as input to the model. Number of images generated=batch_size
  - y: ground truth labels (one-hot encoded) corresponding to X
  """

    def __init__(
        self,
        ground_truth=None,
        input_data=None,
        image_list=None,
        input_size=136,
        num_classes=None,
        batch_size=16,
        to_fit=True,
        shuffle=True,
    ):

        self.ground_truth = ground_truth
        self.input_data = input_data
        self.image_list = image_list
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        if self.to_fit:
            self.on_epoch_end()

    def _data_generator(self, batch_list):
        X = np.zeros((self.batch_size, self.input_size))
        y = np.zeros((self.batch_size, self.num_classes))

        for i, val in enumerate(batch_list):
            X[i] = self.input_data[val]
            y[i] = self.ground_truth[val]
        return X, y

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        batch = self.image_list[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate data
        X, y = self._data_generator(batch)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def normalize_img(self, img):
        return img / img.max()

    def standard_normalize(self, img):
        if img.std() != 0.0:
            img = (img - img.mean()) / (img.std())
            img = np.clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0
        return img

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize_img(self, img, resize_dims):
        return cv2.resize(img, resize_dims)


class ImageGenerator(Sequence):
    """
  Image generator inherited from keras.utils.Sequence to be used for vanilla CNN.
  Args:
  - image_list: list of images filenames
  - label_list: list of labels (0,1,2) corresponding to image_list
  - num_classes: int, number of classes to classify images into
  - batch_size: int, size of batch to be used during training
  - input_shape: tuple, specifying shape that matches with the model's input shape
  - num_channels: int, number of channels in input image. If 1, image will be converted to grayscale
  - augment: bool, set to True to use data augmentation
  - augmentation: albumentations.core.composition.Compose (albumentation initialized using Compose attribute)
  - shuffle: bool, set to True to shuffle the dataset
  - to_fit: bool, set to True during training, set to False during prediction.
  Returns:
  - X: images to be used as input to the model. Number of images generated=batch_size
  - y: ground truth labels (one-hot encoded) corresponding to X
  """

    def __init__(
        self,
        df=None,
        image_dir=None,
        image_list=None,
        num_classes=None,
        batch_size=16,
        input_shape=(240, 320),
        num_channels=3,
        augment=False,
        to_fit=True,
        shuffle=True,
        image_format=None,
        augmentation=None,
    ):

        self.df = df
        self.image_dir = image_dir
        self.image_list = image_list
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.augment = augment
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.image_format = image_format
        self.self_supervised = False

        if self.to_fit:
            self.on_epoch_end()

    def _data_generator(self, batch_indices):
        X = np.zeros(
            (
                self.batch_size,
                self.input_shape[0],
                self.input_shape[1],
                self.num_channels,
            )
        )
        y = np.zeros((self.batch_size, self.num_classes))

        for i, val in enumerate(batch_indices):
            img = cv2.imread(
                "{}.{}".format(
                    os.path.join(self.image_dir, self.image_list[val]),
                    self.image_format,
                )
            )
            if self.augment:
                augmented = self.augmentation(image=img)
                img = augmented["image"]
            img = self.resize_img(
                img, resize_dims=(self.input_shape[1], self.input_shape[0])
            )
            # if self.num_channels<3:
            # img = self.grayscale(img)
            # img = self.normalize_img(img)
            img = img / 255
            # replace any NaNs by 1
            img = np.nan_to_num(img, nan=np.float64(1.0))
            if self.num_channels < 3:
                img = np.reshape(img, (img.shape[0], img.shape[1], self.num_channels))
            X[i] = img

            label = (
                self.df[self.df["Path"] == self.image_list[val]]
                .iloc[:, 1:]
                .to_numpy()
                .flatten()
            )
            # one-hot encoding of labels using Keras
            # label = to_categorical(label, num_classes=self.num_classes)
            y[i] = label

        return X, y

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X, y = self._data_generator(indices)
        if self.self_supervised:
            return X, X
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indices)

    def normalize_img(self, img):
        return img / img.max()

    def standard_normalize(self, img):
        if img.std() != 0.0:
            img = (img - img.mean()) / (img.std())
            img = np.clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0
        return img

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize_img(self, img, resize_dims):
        return cv2.resize(img, resize_dims)
