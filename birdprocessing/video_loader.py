import os
import sys
import re

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
from collections import defaultdict
from colorama import Fore, Style

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads,
                 device_id, files):
        super(VideoReaderPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12)
        self.reader = ops.VideoReader(
            device="gpu", filenames=files,
            sequence_length=sequence_length, normalized=False,
            random_shuffle=False, image_type=types.RGB,
            dtype=types.UINT8, initial_fill=2*sequence_length)
        self.transpose = ops.Transpose(device="gpu", perm=[0, 3, 1, 2])

    def define_graph(self):
        output = self.transpose(self.reader(name="Reader"))
        return output

    
class View():
    def __init__(self, idx, video_file_name, ts_file_name, batch_size,
                 sequence_length):
        self.idx = idx
        self.init_time_stamps(ts_file_name, video_file_name)
        self.pipeline = VideoReaderPipeline(
            batch_size=batch_size, sequence_length=sequence_length,
            num_threads=1, device_id=0,
            files=(video_file_name))
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(
            self.pipeline, ["data"], self.epoch_size, auto_reset=True)

    def number_of_frames(self):
        return len(self.ts)

    def init_time_stamps(self, ts_fname, v_fname):
        # make dummy pipeline to find out what the
        # number of frames is....
        tmp_pipeline = VideoReaderPipeline(
            batch_size=1, sequence_length=1,
            num_threads=1, device_id=0,
            files=(v_fname))
        tmp_pipeline.build()
        self.number_of_frames = tmp_pipeline.epoch_size("Reader")
        del tmp_pipeline  # hopefully this frees up memory
        with open(ts_fname, 'r') as f:
            cnt_ts = f.read().splitlines()
            self.ts = [t.split(' ')[1] for t in cnt_ts]
            # What if the number of loaded time stamps is greater
            # than the number of frames in the video? We assume that
            # the missing frames all are at the beginning of the video,
            # because e.g. the raw data did not start on a key frame
            num_elim = len(self.ts) - self.number_of_frames
            if num_elim != 0:
                print(Fore.RED, fname, ' WARN: elim ', num_elim, Style.RESET_ALL)
            self.ts = self.ts[num_elim:]

    def good_frames(self, valid_ts):
        """compute list of local frame numbers that have
        valid time stamps"""
        vts = set(valid_ts)
        frames = []
        for fn, t in enumerate(self.ts):
            if t in vts:
                frames.append(fn)
        return frames

    def __iter__(self):
        return self.dali_iterator.__iter__()


class SingleVideoIterator(object):
    def __init__(self, idx, dali_iterator, seq_len, common_frames):
        self.idx = idx
        self.dali_iterator = dali_iterator
        self.seq_len = seq_len
        # common_frames holds the local iterators frame numbers
        # that are valid across all cameras.
        self.common_frames = common_frames
        # frame_cnt is the local frame count, referring to the
        # frame numbers as counted via the dali iterator
        self.frame_cnt = -1
        # 
        self.seq_head = -1
        self.seq  = None

    def __next__(self):
        if len(self.common_frames) == 0:
            raise StopIteration
        # skip all frames that are not common
        data = None
        while self.frame_cnt < self.common_frames[0]:
            data = self.dali_iterator.next()
            data[0]['data'] = data[0]['data'] / 255.0
            self.frame_cnt += 1
        # remove first frame
        self.common_frames.pop(0)
        # print('iter ', self.idx, ' delivers frame: ', self.frame_cnt - 1)
        return data

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def reset(self):
        self.frame_cnt = -1
        self.dali_iterator.reset()


class SynchronizedVideoIterator(object):
    def __init__(self, iterators, ts):
        self.iterators = iterators
        self.ts = ts
        self.frame_cnt = 0

    def __next__(self):
        if self.frame_cnt >= len(self.ts):
            raise StopIteration
        frames = []
        stamp = self.ts[self.frame_cnt]
        for i, it in enumerate(self.iterators):
            iv = it.next()[0]
            iv['time'] = stamp
            iv['frame'] = self.frame_cnt
            frames.append(iv)
        self.frame_cnt += 1
        return frames

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__()

    def __iter__(self):
        return self

    def reset(self):
        for i in self.iterators:
            i.reset()
        self.frame_cnt = 0


class SynchronizedVideoLoader():
    def __init__(self, file_root, sequence_length):
        file_type = 'mp4'
        container_files = sorted([f for f in os.listdir(file_root)
                                  if re.match(r'[0-z].*\.' + file_type, f)])
        print('video files: ', container_files)
        full_path = [file_root + '/' + f for f in container_files]
        all_files = [[f, f.replace('.mp4', '_ts.txt')] for f in full_path]
        self.views = []
        for idx, f in enumerate(all_files):
            view = View(idx, f[0], f[1], batch_size=1,
                        sequence_length=sequence_length)
            print('video %s has %7d frames' % (f[0], view.epoch_size))
            self.views.append(view)
        valid_ts = self.find_common_time_stamps(self.views)
        iterators = [
            SingleVideoIterator(
                v.idx, v.dali_iterator, sequence_length,
                v.good_frames(valid_ts)) for v in self.views]
        self.sync_iterator = SynchronizedVideoIterator(iterators, valid_ts)
        self.epoch_size = len(valid_ts)
        print('synchronized video number of frames: ', self.epoch_size)

    def find_common_time_stamps(self, views):
        cnt = defaultdict(int)
        for v in views:
            for t in v.ts:
                cnt[t] += 1

        valid_ts = sorted([t for t in cnt if cnt[t] == len(views)])
        return valid_ts

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.sync_iterator.__iter__()
