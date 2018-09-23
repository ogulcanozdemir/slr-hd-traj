import numpy as np
import os

from scipy.io import loadmat


def read_skeleton(skeleton_path):
    skeleton_path = os.path.join(skeleton_path, 'skeleton.mat')
    skeleton = loadmat(skeleton_path)
    return skeleton


def find_hand_coordinates(skeleton, scaled=False, dist2wrist=35):
    # left_hand = skeleton['skeleton']['HandRight'][0, 0][:, 7:9]
    # right_hand = skeleton['skeleton']['HandLeft'][0, 0][:, 7:9]
    sk_lwrist = skeleton['skeleton']['WristRight'][0, 0]
    sk_lelbow = skeleton['skeleton']['ElbowRight'][0, 0]
    sk_rwrist = skeleton['skeleton']['WristLeft'][0, 0]
    sk_relbow = skeleton['skeleton']['ElbowLeft'][0, 0]

    if scaled:
        sk_lwrist = sk_lwrist / 3
        sk_lelbow = sk_lelbow / 3
        sk_rwrist = sk_rwrist / 3
        sk_relbow = sk_relbow / 3
        dist2wrist = np.ceil(dist2wrist / 3)

    c_idx_0 = 7
    c_idx_1 = 8

    left_hand = np.zeros((len(sk_lelbow), 2))
    right_hand = np.zeros((len(sk_relbow), 2))
    for f in np.arange(0, len(left_hand)):
        # left hand
        v = np.array([(sk_lwrist[f, c_idx_0] - sk_lelbow[f, c_idx_0]), (sk_lwrist[f, c_idx_1] - sk_lelbow[f, c_idx_1])])
        u = v / np.linalg.norm(v)
        left_hand[f, :] = np.array([(sk_lwrist[f, c_idx_0] + dist2wrist * u[0]), (sk_lwrist[f, c_idx_1] + dist2wrist * u[1])])

        # right hand
        v = np.array([(sk_rwrist[f, c_idx_0] - sk_relbow[f, c_idx_0]), (sk_rwrist[f, c_idx_1] - sk_relbow[f, c_idx_1])])
        u = v / np.linalg.norm(v)
        right_hand[f, :] = np.array([(sk_rwrist[f, c_idx_0] + dist2wrist * u[0]), (sk_rwrist[f, c_idx_1] + dist2wrist * u[1])])

    return left_hand, right_hand

# def find_hand_coordinates(skeleton, use_hand_coords=False, dist2wrist=35, source='kinect', modality='rgb'):
    # if use_hand_coords:
    #     l_hand = skeleton['skeletonHandLeft']['Trapezium']
    #     r_hand = skeleton['skeletonHandRight']['Trapezium']
    # else:
    #     c_idx_0 = None
    #     c_idx_1 = None
    #     if source is not 'kinect':
    #         sk_lwrist = skeleton['skeletonBody']['LWrist'][0, 0]
    #         sk_lelbow = skeleton['skeletonBody']['LElbow'][0, 0]
    #         sk_rwrist = skeleton['skeletonBody']['RWrist'][0, 0]
    #         sk_relbow = skeleton['skeletonBody']['RElbow'][0, 0]
    #         c_idx_0 = 0
    #         c_idx_1 = 1
    #     else:
    #         # mirrored in kinect
    #         sk_lwrist = skeleton['skeleton']['WristRight'][0, 0]
    #         sk_lelbow = skeleton['skeleton']['ElbowRight'][0, 0]
    #         sk_rwrist = skeleton['skeleton']['WristLeft'][0, 0]
    #         sk_relbow = skeleton['skeleton']['ElbowLeft'][0, 0]
    #         if modality is 'rgb':
    #             c_idx_0 = 7
    #             c_idx_1 = 8
    #         else:
    #             c_idx_0 = 9
    #             c_idx_1 = 10
    #
    #     l_hand = np.zeros((len(sk_lelbow), 2))
    #     r_hand = np.zeros((len(sk_relbow), 2))
    #     for f in np.arange(0, len(l_hand)):
    #         # left hand
    #         v = np.array([(sk_lwrist[f, c_idx_0] - sk_lelbow[f, c_idx_0]), (sk_lwrist[f, c_idx_1] - sk_lelbow[f, c_idx_1])])
    #         u = v / np.linalg.norm(v)
    #         l_hand[f, :] = np.array([(sk_lwrist[f, c_idx_0]+dist2wrist*u[0]), (sk_lwrist[f, c_idx_1]+dist2wrist*u[1])])
    #
    #         # right hand
    #         v = np.array([(sk_rwrist[f, c_idx_0] - sk_relbow[f, c_idx_0]), (sk_rwrist[f, c_idx_1] - sk_relbow[f, c_idx_1])])
    #         u = v / np.linalg.norm(v)
    #         r_hand[f, :] = np.array([(sk_rwrist[f, c_idx_0]+dist2wrist*u[0]), (sk_rwrist[f, c_idx_1]+dist2wrist*u[1])])