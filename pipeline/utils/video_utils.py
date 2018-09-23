from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import restoration

import cv2
import numpy as np
import os


def read_video(video_name):
    video_name = os.path.join(video_name, 'color.mp4')
    cap = cv2.VideoCapture(video_name)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    except KeyboardInterrupt:
        cap.release()

    return np.asarray(frames)


def grid_display(frames, no_of_columns=5, figsize=(20, 10), color_map=None):
    fig = plt.figure(figsize=figsize)
    columnn = 0

    for i in np.arange(len(frames)):
        columnn += 1

        if columnn == no_of_columns+1:
            fig = plt.figure(figsize=figsize)
            columnn = 1

        fig.add_subplot(1, no_of_columns, columnn)
        if color_map is 'gray':
            plt.imshow(frames[i], 'gray')
        else:
            plt.imshow(frames[i])
        plt.axis('off')


def grid_display_of(first_frame, flows, visualize=True):
    frames = []
    for fl in flows:
        hsv = np.zeros_like(first_frame)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        frames.append(bgr)

    if visualize:
        grid_display(frames)

    return frames


def im2double(image):
    info = np.iinfo(image.dtype)
    return image.astype(np.float) / info.max


def deblur_image(image):
    psf = np.ones((5, 5)) / 25
    image = conv2(image, psf, 'same')
    image += 0.1 * image.std() * np.random.standard_normal(image.shape)

    deconvolved = restoration.wiener(image, psf, 1, clip=False)
    return deconvolved