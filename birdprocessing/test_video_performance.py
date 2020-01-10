#!/usr/bin/env python3
import argparse
import torch
import time
from birdprocessing.video_loader import SynchronizedVideoLoader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops

#
# Measure performance of nvidia DALI layer. How fast can you go?
# Using ffmpeg with hardware decoding/encoding, we can got at about 450fps
# on a single stream 1920x1200, using this command:
#
# ./ffmpeg -hwaccel cuvid -c:v hevc_cuvid -i /data/2019-06-03/video/video_6.mp4 -c:v h264_nvenc -preset slow /data/tmp/output.mp4
#
#
# With the synchronized video recorder on a Quadro P5000, using a sequence length of 12
# we can get 48.55 fps across 8 streams, with aggregate of 388fps
#


class VideoPipe(Pipeline):
    """This class is not used. Keep code around for future experimentation"""
    def __init__(self, batch_size, seq_len,
                 num_threads, device_id, data, shuffle):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)
        self.input = ops.VideoReader(
            device="gpu", filenames=data, sequence_length=seq_len,
            shard_id=0, num_shards=1,
            random_shuffle=shuffle, initial_fill=2*seq_len)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

def main_test(args):
    """ This code is only kept for future experimentation """
    pipe = VideoPipe(batch_size=args.batch_size, num_threads=16, device_id=0,
                     data=[args.root_dir + "/video_0.mp4",
                           args.root_dir + "/video_1.mp4"],
                     seq_len=args.sequence_length, shuffle=False)
    pipe.build()
    for i in range(args.max_frames):
        t0 = time.time()
        pipe_out = pipe.run()
        s = [args.batch_size, args.sequence_length]
        n_frames = s[0]*s[1]
        dt = time.time() - t0
        print(f'time: {dt} fps: {n_frames/dt}')

def get_args():
    parser = argparse.ArgumentParser(
        description='performance measure video decoding speed',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', '-r', action='store', required=True,
                        help='directory where input video files live.')
    parser.add_argument('--batch_size', '-b', action='store', required=False,
                        default=1, type=int, help='batch size')
    parser.add_argument('--sequence_length', '-s', action='store', required=False,
                        default=1, type=int, help='sequence length')
    parser.add_argument('--max_frames', '-m', action='store', required=False,
                        default=None, type=int, help='max num frames')

    return parser.parse_args()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda available: ' + 'YES' if torch.cuda.is_available() else 'NO')
    res = [1920, 1200]
    loader = SynchronizedVideoLoader(args.root_dir,
                                     sequence_length=args.sequence_length)

    t0 = time.time()
    cnt, agg_cnt = 0, 0
    for i, inputs in enumerate(loader):
        frame_0 = inputs[0]
        s = frame_0['data'].shape
        agg_cnt += len(inputs)
        cnt += 1
        print(i, 'size: ', s, frame_0['time'])
        if args.max_frames is not None and i > args.max_frames:
            print(' reached max number of frames!')
            break
    dt = time.time() - t0

    print('time per frame: %.5fs, fps: %8.2f, agg fps: %8.2f' % (
        dt / cnt, cnt / dt, agg_cnt / dt))

if __name__ == '__main__':
    args = get_args()
    main(args)
