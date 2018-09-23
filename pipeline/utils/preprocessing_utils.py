import numpy as np

from pipeline.utils.video_utils import im2double
from scipy.misc import imresize, imshow, imsave


def get_hand_crop(frame, hand_coords, crop_size, is_flow=False):
    border = int(crop_size / 2)
    cropped_frame = np.zeros((crop_size, crop_size, frame.shape[2]))

    hand_coords = np.round(hand_coords).astype(np.int)

    if not is_flow:
        frame = im2double(frame[:, :, :])

    x0 = hand_coords[0] - border
    y0 = hand_coords[1] - border
    x1 = hand_coords[0] + border
    y1 = hand_coords[1] + border

    if x0 <= 0:
        x1 = x1 - x0
        x0 = 1

    if y0 <= 0:
        y1 = y1 - y0
        y0 = 1

    offset_x = np.mod(len(np.arange(x0, x1)), crop_size)
    offset_y = np.mod(len(np.arange(y0, y1)), crop_size)

    if y1 > frame.shape[0]:
        diff = y1 - frame.shape[0]
        cropped_frame[0:y1-y0-diff-offset_y, 0:x1-x0-offset_x, :] = frame[y0:y1-offset_y, x0:x1-offset_x, :]
        cropped_frame[y1-y0:crop_size, :] = 0
    else:
        cropped_frame = frame[y0:y1-offset_y, x0:x1-offset_x, :]

    return cropped_frame


def resize_frames(frames, target_size):
    resized_frames = []

    for f in frames:
        resized_frames.append(imresize(f, target_size))

    return np.asarray(resized_frames)


def get_hand_crop_from_video(frames, skeleton, crop_size=60, resize=None, is_flow=False):
    cropped_frames = np.zeros((frames.shape[0], crop_size, crop_size, frames.shape[3]))

    for f in np.arange(0, frames.shape[0]):
        cropped_frames[f, :, :, :] = get_hand_crop(frames[f], skeleton[f, :], crop_size, is_flow)

    if resize is not None:
        return resize_frames(cropped_frames, resize)

    return cropped_frames