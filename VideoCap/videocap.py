###
# modified from
# https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/blob/master/macOS%20recording%20codes/Real%20time%20webcam%20face%20detection
###

import os
import cv2
import face_recognition
import argparse
import asyncio


class RecordFromWebCam:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def __enter__(self):
        self.video_capture = cv2.VideoCapture(0)

        # Output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Need to scale to this height and width to match Affectiva input
        target_width = 320
        target_height = 240
        self.target_shape = (target_height, target_width)

        frame_width = int(self.video_capture.get(3))
        frame_height = int(self.video_capture.get(4))
        self.frame_shape = (frame_height, frame_width)

        video_fps = 14

        self.out = cv2.VideoWriter()
        self.out.open(
            self.output_dir, fourcc, video_fps, (target_width, target_height), True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Cleaning up...")
        self.video_capture.release()
        self.out.release()
        cv2.destroyAllWindows()

    def resize_frame(self, frame):
        frame_remainder = tuple(
            t1 % t2 for t1, t2 in zip(self.frame_shape, self.target_shape)
        )
        crop_width = frame_remainder[0] // 2
        crop_height = frame_remainder[1] // 2
        height, width = self.frame_shape
        target_height, target_width = self.target_shape
        frame = frame[
            crop_height : height - crop_height, crop_width : width - crop_width
        ]
        frame = cv2.resize(frame, (target_width, target_height))
        return frame

    def run(self):
        while True:
            # Grab a single frame of video
            ret, frame = self.video_capture.read()
            frame = self.resize_frame(frame)

            # Display the resulting image
            cv2.imshow("Video", frame)

            # Record
            self.out.write(frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


async def main(args):
    with RecordFromWebCam(args.output) as rec:
        rec.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_output = os.path.join(os.getcwd(), "vidcap_output.avi")
    parser.add_argument("-o", "--output", type=str, default=default_output)
    args = parser.parse_args()
    asyncio.run(main(args))
