import numpy as np
import itertools
import os

A_DIM = 6
BITRATE_LEVELS = 6


class VideoLoader:
    def __init__(self, filename):
        self.filename = filename
        self.video_size = {}  # in bytes
        self.vmaf_size = {}
        self.VIDEO_BIT_RATE = [375, 750, 1050, 1750, 3000, 4300]  # Kbps
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            self.vmaf_size[bitrate] = []
            VIDEO_SIZE_FILE = None
            for p in os.listdir(self.filename + '/size/'):
                if str(self.VIDEO_BIT_RATE[bitrate]) in p:
                    VIDEO_SIZE_FILE = p
                    break
            with open(self.filename + '/size/' + VIDEO_SIZE_FILE) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))
            with open(self.filename + '/vmaf/' + VIDEO_SIZE_FILE) as f:
                for line in f:
                    self.vmaf_size[bitrate].append(float(line))

    def get_video_size(self):
        return self.video_size

    def get_vmaf_size(self):
        return self.vmaf_size

    def get_chunk_count(self):
        return len(self.video_size[0])
