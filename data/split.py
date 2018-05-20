from collections import OrderedDict

import numpy as np


def load_split(split_path, key_frames=False):
    _split = {}
    with open(split_path, 'r') as f:
        for line in zip(f):
            line_split = line[0][:-1].split(' ')
            video_split = line_split[0].split('/')
            video_label = int(video_split[-2])

            if key_frames:
                key_frames = np.asarray(line_split[1].split(','), dtype=np.int) - 1
                max_kf = np.max(key_frames)
                key_frames[key_frames == max_kf] = max_kf - 1
                _split[line_split[0]] = (video_label, key_frames)
            else:
                _split[line_split[0]] = (video_label)
        f.close()

    return OrderedDict(sorted(_split.items()))