import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
import os
import argparse
import cv2


def save_frame(frame, output_dir, subdir_name, timestep, save_format):
    """
  Saves one frame at a time.
  """
    if save_format == "npy":
        np.save(
            "{}.npy".format(os.path.join(output_dir, subdir_name, str(int(timestep)))),
            frame,
        )
    else:
        cv2.imwrite(
            "{}.{}".format(
                os.path.join(output_dir, subdir_name, str(int(timestep))), save_format
            ),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


def extract_frames(csv_dir, video_dir, output_dir, save_format):
    """
  Extracts frames from video using csv data. It then saves those frames in separate directories
  inside `output_dir` with the names of the directories being the name of the videos and the 
  frames named time_x where x is time in seconds.

  Args:
  - csv_dir: str, path to directory containing the formatted csv files with columns Time and Labels.
  - video_dir: str, path to directory containing the labeled flash videos.
  - output_dir: str, path to directory where sub-directories containing frames will be saved. This will be created if it does not exist.
  - save_format: str, file format for the frames.

  Returns:
  Saves images to directory, does not return a value.
  """
    if not os.path.isdir(output_dir):
        print(
            "Destination directory {} does not exist, creating one now...".format(
                output_dir
            )
        )
        os.makedirs(output_dir)

    csv_files = [file for file in os.listdir(csv_dir) if file.endswith(".csv")]
    saveCount = 0
    print("Found {} csv files in {}".format(len(csv_files), csv_dir))
    print("Starting extraction of frames ...\n")
    for csv in csv_files:
        df = pd.read_csv(os.path.join(csv_dir, csv))
        timelist = df["Time"].to_numpy(dtype="float")
        # Name of sub-directory given by common characters in filenames shared by csv and video files.
        subdir_name = os.path.splitext(csv)[0][:-6]
        # create a directory inside output_dir if it doesn't already exist
        if not os.path.isdir(os.path.join(output_dir, subdir_name)):
            os.makedirs(os.path.join(output_dir, subdir_name))
        # filename of the video has to correspond to the csv file.
        video_filename = subdir_name + ".flv"
        clip = VideoFileClip(os.path.join(video_dir, video_filename))
        for timestep in timelist:
            # extract a frame at the specified timestep
            frame = clip.get_frame(timestep)
            # save to file
            save_frame(frame, output_dir, subdir_name, timestep, save_format)
            saveCount += 1
    print("Saved a total of {} images".format(saveCount))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir",
        default="labels/",
        type=str,
        help="Path to the formatted csv files directory",
    )
    parser.add_argument(
        "--video_dir",
        default="save/",
        type=str,
        help="Path to the directory containing the video files",
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        type=str,
        help="Directory where frames will be saved. If it doesn't exist, one will be created",
    )
    parser.add_argument(
        "--save_format",
        default="png",
        type=str,
        help="Type of file to be saved. Choose among png, jpg, jpeg or npy for ideal results",
    )
    args = parser.parse_args()
    extract_frames(
        csv_dir=args.csv_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        save_format=args.save_format,
    )
    print("Finished extracting and saving frames")
