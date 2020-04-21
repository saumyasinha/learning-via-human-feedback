###
# modified from
# https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/blob/master/macOS%20recording%20codes/Real%20time%20webcam%20face%20detection
###

import argparse
import asyncio
import os

import cv2
import face_recognition


class RecordFromWebCam:
    def __init__(self, uuid, output_dir):
        self.uuid = uuid
        self.output_dir = output_dir
        self.video_output = os.path.join(output_dir, f"{self.uuid}.avi")
        self.frame_output = os.path.join(output_dir, str(self.uuid))
        os.makedirs(self.frame_output, exist_ok=True)

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
            self.video_output, fourcc, video_fps, (target_width, target_height), True,
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

    def get_frame(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()
        frame = self.resize_frame(frame)
        return frame

    def show_frame(self, frame):
        cv2.imshow("Video", frame)

    def write_frame(self, frame):
        self.out.write(frame)

    def write_frame_image(self, frame, timestamp):
        path = os.path.join(self.frame_output, f"{timestamp}.png")
        cv2.imwrite(path, frame)

    def run(self):
        while True:
            # Grab a single frame of video
            frame = self.get_frame()

            # Display the resulting image
            self.show_frame(frame)

            # Record
            self.write_frame(frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


async def main(args):
    with RecordFromWebCam(args.output) as rec:
        rec.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_output = os.getcwd()
    parser.add_argument("-o", "--output", type=str, default=default_output)
    args = parser.parse_args()
    asyncio.run(main(args))
