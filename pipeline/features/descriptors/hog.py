from skimage.feature import hog as sk_hog
from scipy.misc import imsave

import cv2
import numpy as np


# cell_size and block_size parameters should be tuple
def extract_hog_from_frame(frame, num_orientations, cell_size, block_size, visualize=True, resize=False):
    frame = frame.astype(np.float32)

    # check if frame is grayscale (for flow channels [we assume that flow channels represented as one-dimensional grayscale images])
    if frame.ndim == 3 and frame.shape[2] != 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if resize:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    if visualize:
        fd, hog_image = sk_hog(frame, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, block_norm='L2', visualise=visualize, feature_vector=True)
        return fd, hog_image
    else:
        fd = sk_hog(frame, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, block_norm='L2', visualise=visualize, feature_vector=True)
        return fd


def extract_hog_from_video(frames, num_orientations=8, cell_size=15, block_size=4, visualize=False, resize=False):
    cell_size = (cell_size, cell_size)
    block_size = (block_size, block_size)

    hog_fds = []
    hog_images = []

    for f in np.arange(0, frames.shape[0]):
        if visualize:
            fd, hog_image = extract_hog_from_frame(frames[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize, resize=resize)
            hog_images.append(hog_image)
        else:
            fd = extract_hog_from_frame(frames[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize, resize=resize)

        hog_fds.append(fd)

    if visualize:
        return np.asarray(hog_fds), hog_images
    else:
        return np.asarray(hog_fds)