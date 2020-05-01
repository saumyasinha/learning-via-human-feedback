from keras.models import load_model
import numpy as np
import face_recognition
import cv2


emotion_dict = {
    "Angry": 0,
    "Sad": 5,
    "Neutral": 4,
    "Disgust": 1,
    "Surprise": 6,
    "Fear": 2,
    "Happy": 3,
}
# model = load_model('weights/model_v6_23.hdf5')
"""
## Using pre-trained model from https://github.com/priya-dwivedi/face_and_emotion_detection
"""


def get_face_image(img, resize_dims):

    image = cv2.resize(img, resize_dims)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # face_locations = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.5,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )
    # x, y, w, h = face_locations[0]
    # face_image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_locations = face_recognition.face_locations(image, model="cnn")
    print(face_locations)
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    print(face_image.shape)

    return img


def postprocess(img, resize_dims=(48, 48)):

    face_image = cv2.resize(img, resize_dims)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(
        face_image, [1, face_image.shape[0], face_image.shape[1], 1]
    )

    return face_image


def emotion_prediction_on_frame(
    frame, model, model_path=None, initial_resize_dims=None
):

    # read the image and preprocess it
    face_image = get_face_image(frame, resize_dims=(150, 150))
    face_image = postprocess(face_image)

    predicted_class = np.argmax(model.predict(face_image))

    label_map = dict((v, k) for k, v in emotion_dict.items())
    predicted_label = label_map[predicted_class]
    print(predicted_label)

    if predicted_class in [3, 6]:
        return 1

    if predicted_class in [5, 1]:
        return -1

    if predicted_class in [2, 4, 0]:
        return 0
