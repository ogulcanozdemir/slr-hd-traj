from scipy import sqrt
from pipeline.features.descriptors.hog import extract_hog_from_frame

import numpy as np


def mbh(flow, orientations=9, pixels_per_cell=(32,32), cells_per_block=(4,4), visualise=False, normalise=False):

    flow = np.atleast_2d(flow)

    # normalise with gamma compression
    if normalise:
        flow = sqrt(flow)

    # compute first order image gradients
    if flow.dtype.kind == 'u':
        flow = flow.astype('float')

    fx = np.zeros(flow.shape[:2])
    fy = np.zeros(flow.shape[:2])

    fx = flow[:, :, 1]
    fy = flow[:, :, 0]

    if visualise:
        mbhx, mbhx_image = extract_hog_from_frame(fx, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise)
        mbhy, mbhy_image = extract_hog_from_frame(fy, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise)
        return mbhx, mbhy, mbhx_image, mbhy_image
    else:
        mbhx = extract_hog_from_frame(fx, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise)
        mbhy = extract_hog_from_frame(fy, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise)
        return mbhx, mbhy


def extract_mbh_from_flow(flow, num_orientations=8, cell_size=(32,32), block_size=(4,4), visualize=True, normalize=False):
    if visualize is True:
        mbhx, mbhy, mbhx_image, mbhy_image = mbh(flow, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, visualise=visualize, normalise=normalize)
        return mbhx, mbhy, mbhx_image, mbhy_image
    else:
        mbhx, mbhy = mbh(flow, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, visualise=visualize, normalise=normalize)
        return mbhx, mbhy


def extract_mbh_from_flow_seq(flows, num_orientations=8, cell_size=(32,32), block_size=(4,4), visualize=True, normalize=False):
    mbhx_fds = []
    mbhy_fds = []
    mbhx_images = []
    mbhy_images = []

    for f in np.arange(0, flows.shape[0]):
        if visualize:
            mbhx, mbhy, mbhx_image, mbhy_image = extract_mbh_from_flow(flows[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize, normalize=normalize)
            mbhx_images.append(mbhx_image)
            mbhy_images.append(mbhy_image)
        else:
            mbhx, mbhy = extract_mbh_from_flow(flows[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize, normalize=normalize)

        mbhx_fds.append(mbhx)
        mbhy_fds.append(mbhy)

    if visualize:
        return mbhx_fds, mbhy_fds, mbhx_images, mbhy_images
    else:
        return mbhx_fds, mbhy_fds