#!/usr/bin/env python3

import copy
import numpy as np
import cv2
import yaml
import argparse


def matrix_to_tf(T):
    rvec, _ = cv2.Rodrigues(T[0:3, 0:3])
    tvec = T[0:3, 3]
    return rvec[:, 0], tvec


class Camera:
    def __init__(self, name, calib):
        self.name = name
        tf = calib['T_cam_body']
        self.rvec, self.tvec = matrix_to_tf(np.asarray(tf))
        K = np.asarray(calib['intrinsics'])
        self.K = np.array([[K[0], 0.0,  K[2]],
                           [0.0,  K[1], K[3]],
                           [0.0,  0.0,  1.0]])
        self.dist = np.array(calib['distortion_coeffs']).astype('float')
        self.dist_model = calib['distortion_model']


class Projector:
    def __init__(self, calib):
        self.cameras = []
        calib_a = [x for x in calib.items()]
        for i, cam in enumerate(calib_a):
            self.cameras.append(Camera(cam[0], cam[1]))
        print('read calib for %d cameras' % len(self.cameras))

    def project(self, p3d):
        img_pts = np.zeros((len(self.cameras), p3d.shape[0], 2))
        for i, c in enumerate(self.cameras):
            proj, _ = cv2.projectPoints(p3d, c.rvec, c.tvec, c.K, c.dist)
            img_pts[i, :, :] = proj[:, 0, :]
        return img_pts


def read_yaml(filename):
    with open(filename, 'r') as y:
        try:
            return yaml.load(y, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tests the projector.')
    parser.add_argument('--calib_file', '-c', action='store', required=True,
                        help='Name of calibration file.')

    args = parser.parse_args()
    calib = read_yaml(args.calib_file)
    p3d = np.array(((3.0, 1.2, 1.3),
                    (3.2, 1.5, 1.5),
                    (2.9, 1.4, 1.6),
                    (3.1, 1.2, 1.4)))
    proj = Projector(calib)
    img_pts = proj.project(p3d)
    print('img_pts:\n', img_pts)
