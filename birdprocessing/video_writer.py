#!/usr/bin/env python3
import argparse
import torch
import time
import subprocess as sp
import os
from threading import Thread
import queue
from queue import Queue


class VideoFile(Thread):
    def __init__(self, queue, slot, output_dir, res, fps):
        Thread.__init__(self)
        self.queue = queue
        self.total_bytes = 0
        self.slot = slot
        self.frame_num = 0
        self.dt = 0
        self.dt_idle = 0
        fn_base = output_dir + "/video_" + str(slot)
        vid_fn = fn_base + ".mp4"
        ts_fn = fn_base + "_ts.txt"
        size = f'{res[0]}x{res[1]}'
        FNULL = open(os.devnull, 'w')
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-s', size, '-pix_fmt', 'bgr24',
               '-framerate', f'{fps}', '-i', '-',
               '-an',
               '-vsync', '0',
               '-vcodec', 'hevc_nvenc',
               '-preset', 'slow', vid_fn]
        # cmd2 = ['cat', '-']
        self.video_file = sp.Popen(cmd, bufsize=1000000, stdin=sp.PIPE,
                                   stdout=FNULL, stderr=FNULL)
        self.ts_file = open(ts_fn, 'w')
        self.keep_running = True

    def run(self):
        while self.keep_running:
            try:
                t00 = time.time()
                frame, ts = self.queue.get(timeout=1.0)
                t0 = time.time()
                self.queue.task_done()
                self.ts_file.write(str(self.frame_num) + ' ' + ts)
                self.frame_num += 1
                if frame.shape[1] == 3:
                    b = frame[self.slot, :, :, :].permute(
                        1, 2, 0).cpu().numpy().squeeze().tobytes()
                    # b = frame[self.slot, :, :, :].cpu().numpy().tobytes()
                    self.total_bytes += len(b)
                    self.video_file.stdin.write(b)
                else:
                    grey = frame[self.slot, :, :, :]
                    rgb = torch.cat((grey, grey, grey), dim=0)
                    b = rgb.permute(1, 2, 0).cpu().numpy().squeeze().tobytes()
                    self.total_bytes += len(b)
                    self.video_file.stdin.write(b)

                self.dt += time.time() - t0
                self.dt_idle += t0 - t00
            except queue.Empty:
                pass  # print('thread ', self.slot, 'got exception empty')

    def enqueue(self, frame, ts):
        self.queue.put((frame, ts))

    def close(self):
        self.keep_running = False
        self.join()
        self.video_file.stdin.close()
        self.ts_file.close()
        if self.dt > 0:
            print(f'video {self.slot} {self.frame_num} has ',
                  f'bw: {self.total_bytes*1e-6/self.dt}')
            print(f'video {self.slot} has idle time: ',
                  f'{self.dt_idle/(self.dt_idle + self.dt)}')
        else:
            print(f'video {self.slot} has no video written!')
            

class VideoWriter():
    def __init__(self, num_videos, output_dir, res,
                 fps=40.0):
        """ resolution is given as [width, height] """
        self.frame_num = 0
        self.queue = Queue()
        self.videos = [VideoFile(self.queue, i, output_dir, res, fps)
                       for i in range(num_videos)]
        for v in self.videos:
            v.start()

    def close(self):
        if self.queue is not None:
            self.queue.join()  # wait for everybody to be finished
            for v in self.videos:
                v.close()
        self.queue = None

    def total_bytes_written(self):
        b = 0
        for v in self.videos:
            b += v.total_bytes
        return b

    def write_frame(self, frame, ts):
        self.queue.join()  # wait for everybody to be finished
        for v in self.videos:
            v.enqueue(frame, ts)

    def __del__(self):
        self.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda available: ' + 'YES' if torch.cuda.is_available() else 'NO')
    res = [1920, 1200]
    writer = VideoWriter(args.num_videos, args.out_dir, res)

    t0 = time.time()
    for i in range(args.max_frames):
        frame = torch.zeros((
            args.num_videos, 3, res[1], res[0]), dtype=torch.uint8).to(device)
        frame[0, 1, 0:(res[1]//3), :] = 255
        frame[0, 2, -(res[1]//3):, :] = 255
        ts = str(i)
        writer.write_frame(frame, ts)
    writer.close()
    dt = time.time() - t0
    fps = args.max_frames / dt
    mbps = writer.total_bytes_written() / dt * 1e-6
    print(f'time per frame: {1.0 / fps} fps: {fps} MB/s: {mbps}')


def get_args():
    parser = argparse.ArgumentParser(
        description='test video writer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_dir', '-r', action='store', required=True,
                        help='directory where output video files go.')
    parser.add_argument('--num_videos', '-n', action='store', required=False,
                        default=8, type=int, help='number of videos')
    parser.add_argument('--max_frames', '-m', action='store', required=False,
                        default=100, type=int, help='number of frames')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
