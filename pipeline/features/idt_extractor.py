from pipeline.features.extractor import Extractor

from time import clock
from os.path import sep

import numpy as np
import os


FEATURE_FIXED_NAME = 'color_scaled_L{}_S2_T{}.features'
FEATURE_FIXED_NAME_TEMP = 'color.features'
FEATURE_TRAIN_PREFIX = 'train_idt'
FEATURE_TEST_PREFIX = 'test_idt'
FEATURE_DICT_POSTFIX = '_dict'
FEATURE_CAT_POSTFIX = '_cat'


class IdtExtractor(Extractor):

    intervals = {
        'hog': range(70, 166),
        'hof': range(166, 274),
        'mbh': range(274, 466)
    }

    def __init__(self, params, data_helper):
        Extractor.__init__(self, params, data_helper)

    def read_training_features(self, type):
        self.prepare_split_features(self.data_helper.training_split, type, save_file=self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type)

    def read_test_features(self, type):
        self.prepare_split_features(self.data_helper.test_split, type, save_file=self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX + '_' + type)

    def prepare_train_features(self, type, return_dict=False):
        self.train_data[type] = self.load_features(self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type, return_dict=return_dict)

    def prepare_test_features(self, type):
        self.test_data[type] = self.load_features(self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX + '_' + type, return_dict=True)

    def prepare_split_features(self, split, type, save_file):
        features_dict = {}
        for video, label in split:
            # feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME.format(self.params.trajectory_length, self.params.temporal_stride))
            feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME_TEMP)
            features_dict[video] = (feature[:, self.intervals[type]], label)

        if save_file:
            self.save_features_to_pickle(save_file + FEATURE_DICT_POSTFIX, features_dict)

        features_cat = []
        for video, f in features_dict.items():
            features_cat.append(f[0])

        if save_file:
            self.save_features_to_pickle(save_file + FEATURE_CAT_POSTFIX, np.vstack(features_cat))

