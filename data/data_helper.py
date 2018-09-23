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

    classmap = {}
    num_classes = None

    def __init__(self, params, exp_type):
        self.data_path = params.data_path
        self.experiment_path = params.experiment_path
        self.training_split_path = params.training_split_path
        self.test_split_path = params.test_split_path
        self.class_ind_path = params.class_ind_path
        self.feature_path = os.path.join(self.experiment_path, exp_type)

        if hasattr(params, 'num_clusters'):
            self.save_path = os.path.join(self.feature_path, 'k{}'.format(params.num_clusters))
        elif hasattr(params, 'nh_lstm'):
            self.save_path = os.path.join(self.feature_path, 'nhlstm{}'.format(params.nh_lstm))
        else:
            self.save_path = self.feature_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

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
                video_label = int(line_split[1])

                if key_frames:
                    key_frames = np.asarray(line_split[1].split(','), dtype=np.int) - 1
                    max_kf = np.max(key_frames)
                    key_frames[key_frames == max_kf] = max_kf - 1
                    _split[line_split[0]] = (video_label, key_frames)
                else:
                    _split[line_split[0]] = (video_label)
            f.close()

        return _split

    @staticmethod
    def load_class_indices(class_ind_path):
        _indices = {}
        with open(class_ind_path, 'r') as f:
            for line in zip(f):
                line_split = line[0][:-1].split(':')
                original_label = int(line_split[0])
                mapped_label = int(line_split[1])
                _indices[original_label] = mapped_label

            f.close()

        return _indices

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

    def __init__(self, params, exp_type):
        temp = None
        if exp_type == 'idt':
            temp = (exp_type + '_l{}_t{}').format(params.trajectory_length, params.temporal_stride)
        elif exp_type == 'idt_hand':
            temp = (exp_type + '_hr{}_l{}_t{}').format(params.hand_radius, params.trajectory_length, params.temporal_stride)
        elif exp_type == 'fhd':
            temp = (exp_type + '_cp{}_nc{}_nbl{}_nbin{}').format(params.crop_size, params.ncell, params.nblock, params.nbins)
        elif exp_type == 'fhd_tf':
            temp = (exp_type + '_lr{}_e{}_b{}_nhlstm{}_dlstm{}').format(params.learning_rate, params.epochs, params.batch_size, params.nh_lstm, params.d_lstm)

        DataHelper.__init__(self, params, temp)
        self.key_frames = params.key_frames == 1 if True else False
        self.prepare_classmap()
        self.prepare_splits()

    def prepare_splits(self):
        DataHelper.prepare_splits(self)

        training_split = self.load_split(self.training_split_path, key_frames=self.key_frames)
        for video, label in training_split.items():
            self.training_split.append((video, self.classmap[label]))
            self.train_labels.append(self.classmap[label])

        test_split = self.load_split(self.test_split_path, key_frames=self.key_frames)
        for video, label in test_split.items():
            self.test_split.append((video, self.classmap[label]))
            self.test_labels.append(self.classmap[label])

    def prepare_classmap(self):
        self.classmap = self.load_class_indices(self.class_ind_path)
        self.num_classes = len(self.classmap)

