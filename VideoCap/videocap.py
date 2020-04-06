###
# modified from
# https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/blob/master/macOS%20recording%20codes/Real%20time%20webcam%20face%20detection
###

import os
import cv2
import face_recognition
import argparse


def capture_webcam(output_dir):
    video_capture = cv2.VideoCapture(0)

    # Output
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # note the lower case

    frame_width = int(video_capture.get(3)) // 2
    frame_height = int(video_capture.get(4)) // 2

    out = cv2.VideoWriter(
        output_dir,
        fourcc,
        10,
        (frame_width, frame_height),
        True,
    )

    # Initialize variables
    face_locations = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "Face", (top + 6, right - 6), font, 0.5, (0, 0, 255), 1)

        # Display the resulting image
        cv2.imshow("Video", frame)

        # Record
        out.write(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


def main(args):
    capture_webcam(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="vidcap_output")
    main(parser.parse_args())
