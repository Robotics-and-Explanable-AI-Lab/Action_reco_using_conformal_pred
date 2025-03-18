#!/usr/bin/env python3

import os
import cv2
import argparse


def extract_frames_from_video(video_path, output_folder, prefix="frame"):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save frames.
        prefix (str): Prefix for frame filenames.
    """
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_filename = os.path.join(output_folder, f"{prefix}_{count:06d}.jpg")
        cv2.imwrite(frame_filename, image)  # Save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(f"✅ Extracted {count} frames from {os.path.basename(video_path)} to {output_folder}")


def process_episodes(input_root, output_root):
    """
    Processes all episode folders to extract frames from each video.

    Args:
        input_root (str): Path where episode folders are located.
        output_root (str): Path where frames will be saved.
    """
    episode_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    print(f"Found {len(episode_folders)} episode folders: {episode_folders}")

    for episode in episode_folders:
        episode_path = os.path.join(input_root, episode)
        video_files = [f for f in os.listdir(episode_path) if f.endswith(".mp4")]

        for video_file in video_files:
            video_path = os.path.join(episode_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(output_root, episode, video_name + "_frames")
            print(f"🚀 Extracting frames from {video_path} to {output_folder} ...")
            extract_frames_from_video(video_path, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in episode folders.")
    parser.add_argument("--input_root", type=str, required=True, help="Path to root folder containing episode folders.")
    parser.add_argument("--output_root", type=str, required=True, help="Path to root folder to save extracted frames.")
    args = parser.parse_args()

    process_episodes(args.input_root, args.output_root)
