from scipy.io import loadmat
from collections import OrderedDict
from os.path import sep

import os
import json
import numpy as np
import time


class DataSample:

    hog = None
    hof = None
    mbh = None


class DataHelper:

    training_split = None
    test_split = None

    train_labels = None
    test_labels = None

    def __init__(self, params):
        self.experiment_id = time.strftime('%Y%m%d-%H%M%S')
        self.data_path = params.data_path
        self.experiment_path = params.experiment_path
        self.training_split_path = params.training_split_path
        self.test_split_path = params.test_split_path
        self.save_path = os.path.join(self.experiment_path, 'experiment-'+self.experiment_id)
        os.makedirs(self.save_path)
        self.feature_path = params.feature_path

    def prepare_splits(self):
        self.training_split = []
        self.test_split = []
        self.train_labels = []
        self.test_labels = []

    @staticmethod
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

    @staticmethod
    def get_concatenated_data(type, descriptors, labels):
        desc_cat = []
        desc_labels = []

        for idx, ds in enumerate(descriptors):
            if len(desc_cat) == 0:
                desc_cat = np.empty((0, ds.shape[1]))
                desc_labels = np.empty((0, 1))

            skip_idx = 0
            if type is 'hog':
                skip_idx = 1
            desc_cat = np.vstack((desc_cat, ds[skip_idx:, :]))
            desc_labels = np.vstack((desc_labels, np.full((ds.shape[0]-skip_idx, 1), labels[idx])))

        return desc_cat, desc_labels


class ToyDataHelper(DataHelper):

    def __init__(self, params):
        DataHelper.__init__(self, params)
        self.key_frames = params.key_frames == 1 if True else False
        self.prepare_splits()

    def prepare_splits(self):
        DataHelper.prepare_splits(self)

        training_split = self.load_split(self.training_split_path, key_frames=self.key_frames)
        for video, label in training_split.items():
            self.training_split.append((video, label))
            self.train_labels.append(label)

        test_split = self.load_split(self.test_split_path, key_frames=self.key_frames)
        for video, label in test_split.items():
            self.test_split.append((video, label))
            self.test_labels.append(label)