#!/usr/bin/env python3
import os
import numpy as np
from numpy.random import randint
import torch
from torch.utils import data
from PIL import Image

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        # Returns the absolute path to the folder containing frames
        return self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    """
    A PyTorch Dataset for loading extracted frames for TSN/TSM.
    This version supports frames named like:
       frame_000086.jpg
    when image_tmpl='{}_frame_{:06d}.jpg' and the frame folder is named 'episode_0/cam_left_wrist_frames'
    """
    def __init__(self, root_path, list_file,
                 num_segments=3,
                 new_length=1,
                 modality='RGB',
                 # Set the template to match your naming convention:
                 image_tmpl='{}_frame_{:06d}.jpg',
                 transform=None,
                 force_grayscale=False,
                 random_shift=True,
                 dense_sample=False,
                 test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        # For RGBDiff, we need one extra frame to compute the difference
        if self.modality == 'RGBDiff':
            self.new_length += 1

        self._parse_list()

    def _parse_list(self):
        """
        Reads the list_file which contains lines in the form:
        <subfolder> <start_frame> <num_frames> <label>
        and converts the subfolder to an absolute path using the provided root_path.
        This forces the path to be relative to your dataset root.
        """
        with open(self.list_file, 'r') as f:
            lines = [x.strip().split(' ') for x in f if x.strip()]

        # Force the subfolder path to be relative to the provided root_path
        for x in lines:
            # If the annotation file already has an absolute path, override it
            # Here, we assume the annotation should be interpreted relative to root_path.
            # For example, if x[0] is "episode_0/cam_left_wrist_frames", then:
            x[0] = os.path.join(self.root_path, os.path.normpath(x[0]))
        self.video_list = [VideoRecord(x) for x in lines]


    def _load_image(self, directory, idx):
        # Construct the frame filename (e.g., frame_000086.jpg)
        frame_name = f"frame_{idx:06d}.jpg"
        frame_path = os.path.join(directory, frame_name)

        if not os.path.exists(frame_path):
            print(f"🚨 Missing frame: {frame_path}")
            return []
        return [Image.open(frame_path).convert('RGB')]

    def _sample_indices(self, record):
        """
        Randomly sample 'num_segments' frame indices from the video segment.
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = (np.multiply(list(range(self.num_segments)), average_duration) +
                       randint(average_duration, size=self.num_segments))
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + record.start_frames

    def _get_val_indices(self, record):
        """
        Evenly sample frame indices for validation.
        """
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + record.start_frames

    def _get_test_indices(self, record):
        """
        Evenly sample frame indices for testing.
        """
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            if self.random_shift:
                segment_indices = self._sample_indices(record)
            else:
                segment_indices = self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        """
        For each sampled index, load the corresponding frames.
        """
        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                if seg_imgs:
                    images.extend(seg_imgs)
                elif images:
                    # Duplicate the last valid frame if needed.
                    images.append(images[-1])
            if p < record.start_frames + record.num_frames - 1:
                p += 1
        if not images:
            raise ValueError(f"No valid frames found for record: {record.path}")
        
        process_data = self.transform(images) if self.transform else images
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
