import numpy as np
from keras.models import load_model
from utils.utils import ImageGenerator
import cv2
import pandas as pd
import argparse

def preprocess(img, resize_dims):
  generator = ImageGenerator(to_fit=False)
  img = generator.normalize_img(img)
  img = generator.resize_img(img,resize_dims)
  img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
  return img

def prediction(img_path, model_path, classes, resize_dims):
  # read the image and preprocess it
  img = cv2.imread(img_path)
  img = preprocess(img, resize_dims)
  # load the model and predict
  model = load_model(model_path)
  predictions = model.predict(img).flatten()
  # match the predictions to the classes
  print('Predicted class probabilities:\n {}'.format(dict(zip(classes,list(predictions)))))
  return dict(zip(classes,list(predictions)))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default='weights/model.h5', type=str, help="Path to the trained model")
  parser.add_argument("--image_path", default=None, type=str, help="Path to the image")
  parser.add_argument("--resize_dims", nargs='+', default=None, type=int, help="Resize dimensions")
  args = parser.parse_args()

  df = pd.read_csv('master.csv')
  classes = df.columns[1:].to_list()
  preds = prediction(img_path=args.image_path, model_path=args.model_path, classes=classes, resize_dims=tuple(args.resize_dims))
