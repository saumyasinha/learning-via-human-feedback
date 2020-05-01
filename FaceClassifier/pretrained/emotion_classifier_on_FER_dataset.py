####### Not working as of now, FER dataset images are too small to obtain facial landmark points!!!########

from sklearn.ensemble import RandomForestClassifier
from FaceClassifier.utils.utils import ImageGenerator
import numpy as np
import dlib
import cv2
import random
import glob
import math
import os
import csv
import argparse
import imageio

emotions = ["Disgust", "Angry", "Fear", "Surprise", "Happy", "Neutral", "Sad"]
# p = pre-treined model path
p = "weights/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def load_data(args):
    w, h = 48, 48
    image = np.zeros((h, w), dtype=np.uint8)
    id = 1
    with open(args.file, "r") as csvfile:
        datareader = csv.reader(csvfile, delimiter=",")
        headers = next(datareader)
        print(headers)
        for row in datareader:
            emotion = row[0]
            pixels = list(map(int, row[1].split()))
            usage = row[2]

            pixels_array = np.asarray(pixels)

            image = pixels_array.reshape(w, h)
            image_folder = os.path.join(args.output, emotion)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            image_file = os.path.join(image_folder, str(id) + ".jpg")
            img_uint8 = image.astype(np.uint8)
            imageio.imwrite(image_file, img_uint8)
            id += 1
            if id % 100 == 0:
                print("Processed {} images".format(id))

    print("Finished processing {} images".format(id))


def extract_from_file(emotion):
    print("./images/%s/*" % emotion)
    ind = emotions.index(emotion)
    ## give the path of the loaded image dataset here
    files = glob.glob("/Users/saumya/Desktop//ferImages/%s/*" % ind)
    print(len(files))

    random.shuffle(files)
    train = files[: int(len(files) * 0.8)]  # get first 80% of file list
    valid = files[-int(len(files) * 0.2) :]  # get last 20% of file list

    return train, valid


def features_on_landmark_points(landmark_points):
    # detections = detector(image, 1)

    xlist = []
    ylist = []

    for i in range(1, 68):  # Store X and Y coordinates in two lists
        xlist.append(float(landmark_points[i, 0]))
        ylist.append(float(landmark_points[i, 1]))
        # xlist.append(float(shape.part(i).x))
        # ylist.append(float(shape.part(i).y))

    xmean = np.mean(xlist)  # Get the mean of both axes to determine centre of gravity
    ymean = np.mean(ylist)
    xcentral = [
        (x - xmean) for x in xlist
    ]  # get distance between each point and the central point in both axes
    ycentral = [(y - ymean) for y in ylist]

    if (
        xlist[26] == xlist[29]
    ):  # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
        anglenose = 0
    else:
        anglenose = int(
            math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi
        )

    if anglenose < 0:
        anglenose += 90
    else:
        anglenose -= 90

    landmarks_vectorised = []
    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
        landmarks_vectorised.append(x)
        landmarks_vectorised.append(y)
        meannp = np.asarray((ymean, xmean))
        coornp = np.asarray((z, w))
        dist = np.linalg.norm(coornp - meannp)
        anglerelative = (
            math.atan((z - ymean) / (w - xmean)) * 180 / math.pi
        ) - anglenose
        landmarks_vectorised.append(dist)
        landmarks_vectorised.append(anglerelative)


def build_dataset():
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
    for emotion in emotions:
        training, validation = extract_from_file(emotion)
        for item in training:
            image = cv2.imread(item)
            generator = ImageGenerator(to_fit=False)
            landmark_points = generator.get_landmark_points(image, detector, predictor)
            if landmark_points != "error":
                features = features_on_landmark_points(landmark_points)
                train_data.append(features)
                train_labels.append(emotions.index(emotion))

        for item in validation:
            image = cv2.imread(item)
            generator = ImageGenerator(to_fit=False)
            landmark_points = generator.get_landmark_points(image, detector, predictor)
            if landmark_points != "error":
                features = features_on_landmark_points(landmark_points)
                valid_data.append(features)
                valid_labels.append(emotions.index(emotion))

    return train_data, train_labels, valid_data, valid_labels


def train(clf):
    X_train, y_train, X_valid, y_valid = build_dataset()

    print("training model")
    print(X_train)
    clf.fit(X_train, y_train)

    score = clf.score(X_valid, y_valid)

    print("Validation Accuracy:" + str(score))


if __name__ == "__main__":

    ## Uncomment the following for the first time when loading dataset
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--file', required=True, help="path of the fer2013.csv file")
    # parser.add_argument('-o', '--output', required=True, help="path of the output directory")
    #
    # args = parser.parse_args()

    # load_data(args)

    ## choose your classifier
    clf = RandomForestClassifier(min_samples_leaf=20)

    ## train model
    train(clf)
