from pipeline.features.fhd_extractor import FhdExtractor

from data.data_helper import ToyDataHelper
from parameter_parser import ParameterParser
from pipeline.utils.video_utils import grid_display_of, grid_display

import os
import cv2
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt


def turn_image_to_rgb(first_frame, fl):
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == '__main__':
    exp_type = 'fhd'
    params = ParameterParser(exp_type).params
    extract = True

    """ Read Dataset Parameters """
    data_helper = ToyDataHelper(params, exp_type=exp_type)

    """ Initialize Extractor """
    extractor = FhdExtractor(params, data_helper)

    _class = '1'
    _video = '050_00036'

    video_path = os.path.join(os.path.join(data_helper.data_path, _class, _video))
    frames = extractor.get_frames_from_video(video_path)

    plt.axis("off")
    plt.imshow(frames[30])
    plt.show()

    l_hand, r_hand = extractor.get_hand_coordinates_from_skeleton(video_path, resize=False)

    flow = extractor.get_optical_flow_from_video_frames(frames)

    plt.axis("off")
    plt.imshow(turn_image_to_rgb(frames[0], flow[31]))
    plt.show()


    # prepare hand crops
    l_hand_cropped_frames = extractor.get_hand_crops_from_frames(frames, l_hand)
    r_hand_cropped_frames = extractor.get_hand_crops_from_frames(frames, r_hand)
    l_hand_cropped_flow = extractor.get_hand_crops_from_frames(flow, l_hand, _is_flow=True)
    r_hand_cropped_flow = extractor.get_hand_crops_from_frames(flow, r_hand, _is_flow=True)

    grid_display_of(frames[0], flow, visualize=True)

    print(1)