import numpy as np

from pipeline.utils.video_utils import im2double
from scipy.misc import imresize


def get_hand_crop_from_frame(frame, hand_coords, border=140, background=True, depth_mask=None):
    cropped_frame = np.zeros((2*border, 2*border, 3))
    cropped_mask = np.zeros((2*border, 2*border))

    hand_coords = np.round(hand_coords).astype(np.int)
    # if hand_coords[0]-border < frame.shape[1] and hand_coords[1]-border < frame.shape[0] and hand_coords[0]+border < frame.shape[1] and hand_coords[1]+border < frame.shape[0]:
    frame = im2double(frame[:, :, :])

    x0 = hand_coords[0]-border
    y0 = hand_coords[1]-border
    x1 = hand_coords[0]+border
    y1 = hand_coords[1]+border

    if x0 <= 0:
        x1 = x1 - x0
        x0 = 1

    if y0 <= 0:
        y1 = y1 - y0
        y0 = 1

    offset_x = np.mod(len(np.arange(x0, x1)), 2*border)
    offset_y = np.mod(len(np.arange(y0, y1)), 2*border)

    if y1 > frame.shape[0]:
        diff = y1 - frame.shape[0]
        cropped_frame[0:y1-y0-diff-offset_y, 0:x1-x0-offset_x, :] = frame[y0:y1-offset_y, x0:x1-offset_x, :]
        cropped_frame[y1-y0:2*border, :] = 0
        if not background:
            cropped_mask[0:y1 - y0 - diff - offset_y, 0:x1 - x0 - offset_x, :] = depth_mask[y0:y1 - offset_y, x0:x1 - offset_x]
            cropped_mask[y1 - y0:2 * border, :] = 0
    else:
        cropped_frame = frame[y0:y1-offset_y, x0:x1-offset_x, :]
        if not background:
            cropped_mask = depth_mask[y0:y1 - offset_y, x0:x1 - offset_x]

    if not background:
        cropped_frame[cropped_mask == 255] = 0

    return cropped_frame


def get_hand_crop_from_flow(flow, hand_coords, border=140):
    cropped_flow = np.zeros((2*border, 2*border, 2))

    hand_coords = np.round(hand_coords).astype(np.int)
    # if hand_coords[0]-border < frame.shape[1] and hand_coords[1]-border < frame.shape[0] and hand_coords[0]+border < frame.shape[1] and hand_coords[1]+border < frame.shape[0]:
    # frame = im2double(frame[:, :, :])

    x0 = hand_coords[0]-border
    y0 = hand_coords[1]-border
    x1 = hand_coords[0]+border
    y1 = hand_coords[1]+border

    if x0 <= 0:
        x1 = x1 - x0
        x0 = 1

    if y0 <= 0:
        y1 = y1 - y0
        y0 = 1

    offset_x = np.mod(len(np.arange(x0, x1)), 2*border)
    offset_y = np.mod(len(np.arange(y0, y1)), 2*border)

    if y1 > flow.shape[0]:
        diff = y1 - flow.shape[0]
        cropped_flow[0:y1-y0-diff-offset_y, 0:x1-x0-offset_x, :] = flow[y0:y1-offset_y, x0:x1-offset_x, :]
        cropped_flow[y1-y0:2*border, :] = 0
    else:
        cropped_flow = flow[y0:y1-offset_y, x0:x1-offset_x, :]
    # cropped_flow = flow[y0:y1-offset_y, x0:x1-offset_x, :]

    return cropped_flow


def resize_frames(frames, target_size):
    resized_frames = []

    for f in frames:
        resized_frames.append(imresize(f, target_size))

    return np.asarray(resized_frames)


def get_hand_crop_from_video(frames, skeleton, resize=None, border=140):
    cropped_frames = np.zeros((frames.shape[0], 2*border, 2*border, 3))

    for f in np.arange(0, frames.shape[0]):
        cropped_frames[f, :, :, :] = get_hand_crop_from_frame(frames[f], skeleton[f, :], border)

    if resize is not None:
        return resize_frames(cropped_frames, resize)

    return cropped_frames


def get_hand_crop_from_flow_seq(flows, skeleton, resize=None, border=140):
    cropped_frames = np.zeros((flows.shape[0], 2*border, 2*border, 2))

    for f in np.arange(0, flows.shape[0]):
        cropped_frames[f, :, :, :] = get_hand_crop_from_flow(flows[f], skeleton[f, :], border)

    if resize is not None:
        return resize_frames(cropped_frames, resize)

    return cropped_frames