import numpy as np
import os
import argparse
import albumentations
from model import vanilla_cnn, landmark_network, vae_network, VAE_LOSS
from sklearn.model_selection import train_test_split
from utils.utils import DataGenerator, ImageGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Model
from functools import partial


def landmark_train(args):
    landmarks = np.load(args.data_file, allow_pickle="TRUE").item()
    facial_action_units = np.load(args.ground_truth_file, allow_pickle="TRUE").item()
    assert (
        facial_action_units.keys() == landmarks.keys()
    ), "Error, ground truth file and input data file haev different data"
    # split data
    X_train, X_val = train_test_split(
        list(facial_action_units.keys()),
        shuffle=True,
        random_state=42,
        test_size=args.test_size,
    )

    # generators for training and validation
    train_gen = DataGenerator(
        ground_truth=facial_action_units,
        input_data=landmarks,
        image_list=X_train,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_gen = DataGenerator(
        ground_truth=facial_action_units,
        input_data=landmarks,
        image_list=X_val,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # build the model
    model = landmark_network(
        input_shape=(args.input_size,),
        num_classes=args.num_classes,
        final_activation_fn="sigmoid",
    )

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


def cnn_train(args, custom_model):
    def prepareData(csv_dir, dropThresh=100):
        list_dfs = []
        dropCols = []
        for i in os.listdir(csv_dir):
            df = pd.read_csv(os.path.join(csv_dir, i))
            # append only the path+AU columns i.e no need for Time and Labels columns
            list_dfs.append(df.iloc[:, 2:])
        # Turn the list into a master dataframe and rearrange the columns
        df = pd.concat(list_dfs)[df.columns[2:]]
        for i in df.columns[1:]:
            if len(df[df[i] == 1]) < dropThresh:
                dropCols.append(i)
        # drop the columns having counts<dropThresh
        df = df.drop(dropCols, axis=1)
        # Also drop any rows that might be all 0s
        df = df[(df.iloc[:, 1:].T != 0).any()].reset_index(drop=True)
        return df

    df = prepareData(args.csv_dir)

    # save to file
    df.to_csv("master.csv", index=False)

    # path column is not part of the classes
    num_classes = len(df.columns[1:])

    # split data into training set and validation set
    X_train, X_val = train_test_split(
        list(df.iloc[:, 0]), shuffle=True, test_size=args.test_size
    )
    validation_filename = f"{os.path.splitext(args.model_name)[0]}.txt"
    with open(validation_filename, "w") as f:
        f.write("Validation images: \n{}".format(X_val))
    print(f"Saved list of validation images in {validation_filename}")

    AUGMENTATIONS = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.3
            ),
            albumentations.GaussNoise(p=0.1, var_limit=3.0),
            albumentations.HueSaturationValue(p=0.25),
            albumentations.Rotate(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
        ]
    )

    # generators for training and validation
    train_gen = ImageGenerator(
        df=df,
        image_dir=args.image_dir,
        image_list=X_train,
        num_classes=num_classes,
        input_shape=(args.input_height, args.input_width),
        batch_size=args.batch_size,
        num_channels=args.input_channels,
        augmentation=AUGMENTATIONS,
        augment=args.augment,
        image_format=args.image_format,
        shuffle=True,
    )

    valid_gen = ImageGenerator(
        df=df,
        image_dir=args.image_dir,
        image_list=X_val,
        num_classes=num_classes,
        input_shape=(args.input_height, args.input_width),
        batch_size=args.batch_size,
        num_channels=args.input_channels,
        augmentation=None,
        augment=False,
        image_format=args.image_format,
        shuffle=True,
    )

    # build the model
    model = custom_model(
        input_shape=(args.input_height, args.input_width, args.input_channels),
        num_classes=num_classes,
        final_activation_fn="sigmoid",
    )

    adam = Adam(learning_rate=args.lr, clipnorm=1.0, clipvalue=0.5)
    loss = "binarycrossentropy"
    if args.model == "vae":
        loss = VAE_LOSS
    model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

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


MODELS = {
    "vanilla": partial(cnn_train, custom_model=vanilla_cnn),
    "vae": partial(cnn_train, custom_model=vae_network),
    "landmark": landmark_train,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="vae")
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
        help="The npy file containing the ground truth AU classes",
    )
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help="The npy file containing the input data",
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
    parser.add_argument("--num_classes", default=15, type=int, help="Number of classes")
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

    model = MODELS[args.model](args)
