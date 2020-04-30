import numpy as np
import os
import argparse
from model import vanilla_cnn, landmark_network
from sklearn.model_selection import train_test_split
from utils.utils import DataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Model


def train(args):
    landmarks = np.load(args.data_file, allow_pickle='TRUE').item()
    facial_action_units = np.load(args.ground_truth_file, allow_pickle='TRUE').item()
    assert facial_action_units.keys() == landmarks.keys(), "Error, ground truth file and input data file haev different data"
    # split data
    X_train, X_val = train_test_split(
        list(facial_action_units.keys()), shuffle=True, random_state=42, test_size=args.test_size)

    # generators for training and validation
    train_gen = DataGenerator(
        ground_truth = facial_action_units,
        input_data = landmarks,
        image_list = X_train,
        num_classes = args.num_classes,
        batch_size = args.batch_size,
        shuffle = True,
    )

    valid_gen = DataGenerator(
        ground_truth = facial_action_units,
        input_data = landmarks,
        image_list = X_val,
        num_classes = args.num_classes,
        batch_size = args.batch_size,
        shuffle = True,
    )

    # build the model
    model = landmark_network(input_shape=(args.input_size,),num_classes=args.num_classes,final_activation_fn='sigmoid')

    print(model.summary())
    adam = Adam(learning_rate=args.lr, clipnorm=1.0, clipvalue=0.5)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, args.model_name), verbose=1, save_best_only=True
    )
    learn_rate = ReduceLROnPlateau(
        monitor="val_loss", factor=0.8, patience=25, verbose=1
    )
    callback_list = [checkpoint, learn_rate]
    history = model.fit_generator(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        verbose=1,
        callbacks=callback_list,
    )
    return history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="weights/",
        type=str,
        help="Directory where model will be stored",
    )
    parser.add_argument(
        "--model_name",
        default="model.h5",
        type=str,
        help="File Name of .h5 file which will contain the weights and saved in model_dir",
    )
    parser.add_argument(
        "--ground_truth_file",
        default=None,
        type=str,
        help="The npy file containing the ground truth AU classes"
    )
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help="The npy file containing the input data"
    )
    parser.add_argument(
        "--input_size", default=136, type=int, help="Input size for the model"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for the model"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate for the model"
    )
    parser.add_argument(
        "--num_classes", default=15, type=int, help="Number of classes"
    )
    parser.add_argument(
        "--epochs", default=500, type=int, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--test_size",
        default=0.20,
        type=float,
        help="Fraction of training image to use for validation during training",
    )

    args = parser.parse_args()

    model = train(args)
