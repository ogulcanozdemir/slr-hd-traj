from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

import cv2
import numpy as np


def calc_optical_flow_from_frame(image_prev, image_next):
    image_prev = cv2.cvtColor(image_prev.astype(np.float32), cv2.COLOR_RGB2GRAY)
    image_next = cv2.cvtColor(image_next.astype(np.float32), cv2.COLOR_RGB2GRAY)

    # get optical flow
    # flow = cv2.calcOpticalFlowFarneback(image_prev, image_next, flow=None, pyr_scale=.5, levels=3, winsize=9, iterations=1, poly_n=3, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = cv2.calcOpticalFlowFarneback(image_prev, image_next, None, pyr_scale=0.5, levels=3, winsize=9, iterations=1, poly_n=3, poly_sigma=1.1, flags=0)
    return flow


def calc_optical_flow_from_video(frames):
    flows = []

    for f in np.arange(0, frames.shape[0]-1):
        flows.append(calc_optical_flow_from_frame(frames[f], frames[f+1]))

    return flows


def extract_hof_from_flow(flow, num_orientations=9, cell_size=(32,32), block_size=(4,4), visualize=True):
    flow = cv2.resize(flow, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    if visualize:
        fd, hof_image = hof(flow, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, visualise=visualize, normalise=False, motion_threshold=1.)
        return fd, hof_image
    else:
        fd = hof(flow, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, visualise=visualize, normalise=False, motion_threshold=1.)
        return fd


def extract_hof_from_flow_seq(flows, num_orientations=9, cell_size=(32,32), block_size=(4,4), visualize=True):
    hof_fds = []
    hof_images = []

    for f in np.arange(0, flows.shape[0]):
        if visualize:
            fd, hof_image = extract_hof_from_flow(flows[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize)
            hof_images.append(hof_image)
        else:
            fd = extract_hof_from_flow(flows[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize)

        hof_fds.append(fd)

    if visualize:
        return hof_fds, hof_images
    else:
        return hof_fds


# L2 norm by default
def hof(flow, orientations=9, pixels_per_cell=(32,32), cells_per_block=(4,4), visualise=False, normalise=False, motion_threshold=1.):

    flow = np.atleast_2d(flow)

    # normalise with gamma compression
    if normalise:
        flow = sqrt(flow)

    # compute first order image gradients
    if flow.dtype.kind == 'u':
        flow = flow.astype('float')

    gx = np.zeros(flow.shape[:2])
    gy = np.zeros(flow.shape[:2])

    gx = flow[:, :, 1]
    gy = flow[:, :, 0]

    # compute gradient magnitudes and orientation
    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / np.pi) % 180

    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))
    n_cellsy = int(np.floor(sy // cy))

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[int(cy / 2):cy * n_cellsy:cy, int(cx / 2):cx * n_cellsx:cx]

    for i in range(orientations-1):
        # isolate orientations in this range
        temp_ori = np.where(orientation < 180  / orientations * (i+1), orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i, temp_ori, -1)

        # select magnitudes for those orientations
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]

    # calculate no-motion bin
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    # now for each cell, compute the histogram
    hof_image = None

    if visualise:
        from skimage import draw

        radius = min(cx, cy) // 2 - 1
        orientations_arr = np.arange(orientations)
        dx_arr = radius * cos(orientations_arr / orientations * np.pi)
        dy_arr = radius * sin(orientations_arr / orientations * np.pi)
        hof_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    rr, cc = draw.line(int(centre[0] - dx),
                                       int(centre[1] + dy),
                                       int(centre[0] + dx),
                                       int(centre[1] - dy))
                    hof_image[rr, cc] += orientation_histogram[y, x, o]

    # block normalisation
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)

    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()