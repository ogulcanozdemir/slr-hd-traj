from scipy import sqrt
from pipeline.features.descriptors.hog import extract_hog_from_frame

import numpy as np


# cell_size and block_size parameters should be tuple
def mbh(flow, orientations, pixels_per_cell, cells_per_block, visualise, normalise, resize=False):
    flow = np.atleast_2d(flow)

    # normalise with gamma compression
    if normalise:
        flow = sqrt(flow)

    # compute first order image gradients
    if flow.dtype.kind == 'u':
        flow = flow.astype('float')

    fx = flow[:, :, 1]
    fy = flow[:, :, 0]

    if visualise:
        mbhx, mbhx_image = extract_hog_from_frame(fx, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise, resize=resize)
        mbhy, mbhy_image = extract_hog_from_frame(fy, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise, resize=resize)
        return mbhx, mbhy, mbhx_image, mbhy_image
    else:
        mbhx = extract_hog_from_frame(fx, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise, resize=resize)
        mbhy = extract_hog_from_frame(fy, num_orientations=orientations, cell_size=pixels_per_cell, block_size=cells_per_block, visualize=visualise, resize=resize)
        return mbhx, mbhy


# cell_size and block_size parameters should be tuple
def extract_mbh_from_flow(flow, num_orientations, cell_size, block_size, visualize, normalize, resize=False):
    if visualize is True:
        mbhx, mbhy, mbhx_image, mbhy_image = mbh(flow, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, visualise=visualize, normalise=normalize, resize=resize)
        return mbhx, mbhy, mbhx_image, mbhy_image
    else:
        mbhx, mbhy = mbh(flow, orientations=num_orientations, pixels_per_cell=cell_size, cells_per_block=block_size, visualise=visualize, normalise=normalize, resize=resize)
        return mbhx, mbhy


def extract_mbh_from_flow_seq(flows, num_orientations=8, cell_size=15, block_size=4, visualize=False, normalize=False, resize=False):
    cell_size = (cell_size, cell_size)
    block_size = (block_size, block_size)

    mbhx_fds = []
    mbhy_fds = []
    mbhx_images = []
    mbhy_images = []

    for f in np.arange(0, flows.shape[0]):
        if visualize:
            mbhx, mbhy, mbhx_image, mbhy_image = extract_mbh_from_flow(flows[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize, normalize=normalize, resize=resize)
            mbhx_images.append(mbhx_image)
            mbhy_images.append(mbhy_image)
        else:
            mbhx, mbhy = extract_mbh_from_flow(flows[f, :, :, :], num_orientations=num_orientations, cell_size=cell_size, block_size=block_size, visualize=visualize, normalize=normalize, resize=resize)

        mbhx_fds.append(mbhx)
        mbhy_fds.append(mbhy)

    if visualize:
        return np.hstack((mbhx_fds, mbhy_fds)), mbhx_images, mbhy_images
    else:
        return np.hstack((mbhx_fds, mbhy_fds))