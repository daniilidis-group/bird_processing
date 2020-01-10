#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse


from video_loader import SynchronizedVideoLoader


# helper function for displaying synchronized video frames
#

def show_synchronized_frames(inputs):
    n_cam = len(inputs)
    columns = n_cam // 2
    rows = n_cam // columns
    gs = gridspec.GridSpec(rows, columns, wspace=0.01, hspace=0.01,
                           left=0, right=1.0, top=0.9)
    num_seq = inputs[0]['data'].shape[1]
    for seq in range(num_seq):
        plt.figure()
        for cam_idx in range(n_cam):
            plt.subplot(gs[cam_idx])
            plt.axis("off")
            img_seq = inputs[cam_idx]['data']
            # the first two indexes are batch and sequence number
            # which are always of size 1
            npimg = img_seq[:, :, :].cpu().numpy().squeeze()
            npimg = npimg / 255.0  # normalize to 0...1
            plt.imshow(np.transpose(npimg, (1, 2, 0)), vmin=0, vmax=1.0)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tests the video loading.')
    parser.add_argument('--root_dir', '-r', action='store', required=True,
                        help='directory where video files live.')
    parser.add_argument('--sequence_length', '-s', action='store', required=False,
                        default=12, help='decoding sequence length.')

    args = parser.parse_args()


    # Instantiate synchronized video loader
    loader = SynchronizedVideoLoader(args.root_dir, sequence_length=args.sequence_length)

    for i, inputs in enumerate(loader):
        frame_0 = inputs[0]
        print(i, 'size: ', frame_0['data'].shape, frame_0['time'])
        show_synchronized_frames(inputs)
