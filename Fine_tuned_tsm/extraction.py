#!/usr/bin/env python3

import os
import h5py
import cv2
import numpy as np
import argparse


def extract_single_hdf5_video(hdf5_path, output_dir, fps=30):
    """
    Extract and save videos from a SINGLE HDF5 file.

    Args:
        hdf5_path (str): Path to the input HDF5 file.
        output_dir (str): Path to store extracted videos.
        fps (int): Frames per second for output videos.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    with h5py.File(hdf5_path, 'r') as f:
        compressed = f.attrs.get('compress', False)
        print(f"Compression status: {compressed}")

        cam_names = list(f['/observations/images'].keys())
        print(f"Found cameras: {cam_names}")

        # ✅ Check for compression length
        compress_len = None
        if compressed and '/compress_len' in f:
            compress_len = f['/compress_len'][()]
        else:
            print(f"⚠️ Compression True but '/compress_len' missing or not used. Handling as uncompressed.")
            compressed = False  # Treat as uncompressed if `/compress_len` is missing

        # 🔹 Process each camera
        for cam_idx, cam_name in enumerate(cam_names):
            print(f"Processing {cam_name}...")

            image_list = []
            padded_compressed_image_list = f[f'/observations/images/{cam_name}']

            # Decompress images if compressed
            if compressed:
                for frame_idx, padded_image in enumerate(padded_compressed_image_list):
                    length = int(compress_len[cam_idx, frame_idx])
                    compressed_image = padded_image[:length]
                    image = cv2.imdecode(compressed_image, 1)  # Decode JPEG
                    image_list.append(image)
            else:
                image_list = padded_compressed_image_list[()]  # Uncompressed

            if len(image_list) == 0:
                print(f"⚠️ No images found for {cam_name}")
                continue

            # Prepare video writer
            height, width, _ = image_list[0].shape
            video_path = os.path.join(output_dir, f"{cam_name}.mp4")
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Write frames to video
            for img in image_list:
                video_writer.write(img)

            video_writer.release()
            print(f"✅ Saved video: {video_path}")

    print(f"\n🎉 All camera videos extracted and stored in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract videos from a single HDF5 file.")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the single .hdf5 file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store extracted videos.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output videos (default=30).")

    args = parser.parse_args()

    extract_single_hdf5_video(args.hdf5_path, args.output_dir, fps=args.fps)
