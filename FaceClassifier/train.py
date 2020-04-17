import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from utils import ImageGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import argparse
import albumentations
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.applications import VGG16, Xception
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from losses import focal_loss
import pandas as pd
from model import create_model

def prepare_data(csv_dir)
  list_dfs = []
  for i in os.listdir(csv_dir):
    df = pd.read_csv(os.path.join(csv_dir,i))
    # append only the path+AU columns i.e no need for Time and Labels columns
    list_dfs.append(df.iloc[:,2:])
  # Turn the list into a master dataframe and rearrange the columns
  return pd.concat(list_dfs)[df.columns[2:]]

def train(args):
  df = prepare_data(args.csv_dir)
  # split data into training set and validation set
  X_train, X_val = train_test_split(list(df.iloc[:,0]), shuffle=True, test_size=args.test_size)

  AUGMENTATIONS = albumentations.Compose([
                  albumentations.HorizontalFlip(p=0.5),
                  albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3),
                  albumentations.GaussNoise(p=0.1, var_limit=3.0),
                  albumentations.HueSaturationValue(p=0.25),
                  albumentations.Rotate(p=0.5),
                  albumentations.ShiftScaleRotate(p=0.5)])

  # generators for training and validation
  train_gen = ImageGenerator(df=df,
                             image_dir=args.image_dir,
                             image_list=X_train, 
                             num_classes=args.num_classes,
                             input_shape=(args.input_height, args.input_width),
                             batch_size=args.batch_size,
                             num_channels=args.input_channels,
                             augmentation=AUGMENTATIONS,
                             augment=False,
                             image_format=args.image_format,
                             shuffle=True)

  valid_gen = ImageGenerator(df=df,
                             image_dir=args.image_dir,
                             image_list=X_val,
                             num_classes=args.num_classes,
                             input_shape=(args.input_height, args.input_width),
                             batch_size=args.batch_size,
                             num_channels=args.input_channels,
                             augmentation=None,
                             augment=False,
                             image_format=args.image_format,
                             shuffle=True)

  # build the model
  model = create_model(input_shape=(args.input_height,args.input_width,args.input_channels),
                       num_classes=args.num_classes, final_activation_fn='softmax')
  print(model.summary())
  # base_model = VGG16(weights=None, include_top=False, input_shape=(args.input_height, args.input_width, args.input_channels))
  # for layer in base.layers[:-4]:
  #  layer.trainable = False
  # model = Sequential()
  # base_model = Xception(input_shape=(args.input_height, args.input_width, args.input_channels), include_top=False)
  # for layer in base_model.layers:
  #    layer.trainable = False
  # x = Flatten()(base_model.output)
  # x = Dense(1024, activation='relu')(x)
  # x = Dropout(0.5)(x)
  # x = base_model.output
  # x = GlobalAveragePooling2D()(x)
  # x = Dense(args.num_classes, activation="softmax")(x)
  # model = Model(base_model.input, x)
  adam = Adam(learning_rate=args.lr, clipnorm=1.0, clipvalue=0.5)
  model.compile(optimizer=adam, loss=focal_loss, metrics=["accuracy"])

  checkpoint = ModelCheckpoint(args.model_name, verbose=1, save_best_only=True)
  learn_rate = ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=10, verbose=1)
  callback_list = [checkpoint, learn_rate]
  history = model.fit_generator(train_gen, validation_data=valid_gen, epochs=args.epochs, verbose=1, callbacks=callback_list)
  return history


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("--model_name", default='model.h5', type=str, help="File Name of .h5 file \
                      which will contain the weights and saved in current directory")
  parser.add_argument("--image_dir", default='imgs/', type=str, help="Directory containing the images in sub-directories")
  parser.add_argument("--image_format", default='png', type=str, help="File format of the images")
  parser.add_argument("--csv_dir", default='csvs/', type=str, help="Directory containing the CSV files \
                      containing the ground truth information along with paths to images")
  parser.add_argument("--input_height", default=240, type=int, help="Input height")
  parser.add_argument("--input_width", default=320, type=int, help="Input width")
  parser.add_argument("--input_channels", default=3, type=int, help="Number of channels in input images")
  parser.add_argument("--num_classes", default=25, type=int, help="Number of classes")
  parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the model")
  parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for the model")
  parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train the model")
  parser.add_argument("--augment", default=False, type=bool, help="Flag, set to True if data augmentation needs to be enabled")
  parser.add_argument("--test_size", default=0.20, type=float, help="Fraction of training image to use for validation during training")

  args = parser.parse_args()

  model = train(args)
