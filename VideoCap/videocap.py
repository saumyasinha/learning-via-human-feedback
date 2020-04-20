###
# modified from
# https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/blob/master/macOS%20recording%20codes/Real%20time%20webcam%20face%20detection
###

import os
import cv2
import face_recognition
import argparse


def resize_frame(frame, frame_shape, target_shape):
    frame_remainder = tuple(t1 % t2 for t1, t2 in zip(frame_shape, target_shape))
    crop_width = frame_remainder[0] // 2
    crop_height = frame_remainder[1] // 2
    height, width = frame_shape
    target_height, target_width = target_shape
    frame = frame[crop_height : height - crop_height, crop_width : width - crop_width]
    frame = cv2.resize(frame, (target_width, target_height))
    return frame


def capture_webcam(output_dir):
    video_capture = cv2.VideoCapture(0)

    # Output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Need to scale to this height and width to match Affectiva input
    target_width = 320
    target_height = 240
    target_shape = (target_height, target_width)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    frame_shape = (frame_height, frame_width)

    print(f"frame (height, width): {frame_shape}")

    video_fps = 14

    out = cv2.VideoWriter()
    out.open(
        output_dir, fourcc, video_fps, (target_width, target_height), True,
    )

    try:
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            frame = resize_frame(
                frame, frame_shape=frame_shape, target_shape=target_shape
            )

            # Display the resulting image
            cv2.imshow("Video", frame)

            # Record
            out.write(frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Cleaning up...")
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()


def main(args):
    capture_webcam(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_output = os.path.join(os.getcwd(), "vidcap_output.avi")
    parser.add_argument("-o", "--output", type=str, default=default_output)
    args = parser.parse_args()
    main(args)
