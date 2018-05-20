from skimage.feature import hog as sk_hog

import cv2
import numpy as np


def extract_hog_from_frame(frame, num_orientations=8, cell_size=(32,32), block_size=(4,4), visualize=True):
    frame = frame.astype(np.float32)

    # check if frame is grayscale
    if frame.ndim == 3 and frame.shape[2] != 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    if visualize:
        fd, hog_image = sk_hog(frame, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, block_norm='L2', visualise=visualize, feature_vector=True)
        return fd, hog_image
    else:
        fd = sk_hog(frame, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, block_norm='L2', visualise=visualize, feature_vector=True)
        return fd


def extract_hog_from_video(frames, num_orientations=8, cell_size=(32,32), block_size=(4,4), visualize=True):
    hog_fds = []
    hog_images = []

    for f in np.arange(0, frames.shape[0]):
        if visualize:
            fd, hog_image = extract_hog_from_frame(frames[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize)
            hog_images.append(hog_image)
        else:
            fd = extract_hog_from_frame(frames[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize)

        hog_fds.append(fd)

    if visualize:
        return hog_fds, hog_images
    else:
        return hog_fds


def hog(frame, n_orientations, cell_size, block_size, visualize=True):
    frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # flow = cv2.calcOpticalFlowFarneback(frame, frame, )

    return