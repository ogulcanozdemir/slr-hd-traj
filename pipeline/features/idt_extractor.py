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

    hog_interval = range(70, 166)
    hof_interval = range(166, 274)
    mbh_interval = range(274, 466)

    def __init__(self, params, data_helper):
        Extractor.__init__(self, params, data_helper)

    def read_features(self, type, is_preprocessing=True):
        self.prepare_split_features(self.data_helper.training_split, type, save_file=self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type)
        self.prepare_split_features(self.data_helper.test_split, type, save_file=self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX + '_' + type)

    def prepare_train_features(self, type, return_dict=False):
        train_features = self.load_features(self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type, return_dict=return_dict)
        if type == 'hog':
            self.train_hog = train_features
        elif type == 'hof':
            self.train_hof = train_features
        else:
            self.train_mbh = train_features
        # train_features = self.load_features(self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type, return_dict=return_dict)
        # if return_dict: # if return_dict is True => training mode, if its False preprocessing mode
        #     for spl in self.data_helper.training_split:
        #         if type == 'hog':
        #             self.train_hog.append((spl[0], spl[1], train_features[spl[0]][0][:, self.hog_interval]))
        #         elif type == 'hof':
        #             self.train_hof.append((spl[0], spl[1], train_features[spl[0]][0][:, self.hof_interval]))
        #         else:
        #             self.train_mbh.append((spl[0], spl[1], train_features[spl[0]][0][:, self.mbh_interval]))
        # else:
        #     if type == 'hog':
        #         self.train_hog = train_features[:, :]
        #     elif type == 'hof':
        #         self.train_hof = train_features[:, :]
        #     else:
        #         self.train_mbh = train_features[:, :]

    def prepare_test_features(self, type):
        test_features = self.load_features(self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX + '_' + type, return_dict=True)
        if type == 'hog':
            self.test_hog = test_features
        elif type == 'hof':
            self.test_hof = test_features
        else:
            self.test_mbh = test_features
        # for spl in self.data_helper.test_split:
        #     if type == 'hog':
        #         self.test_hog.append((spl[0], spl[1], test_features[spl[0]][0][:, self.hog_interval]))
        #     elif type == 'hof':
        #         self.test_hof.append((spl[0], spl[1], test_features[spl[0]][0][:, self.hof_interval]))
        #     else:
        #         self.test_mbh.append((spl[0], spl[1], test_features[spl[0]][0][:, self.mbh_interval]))

    def prepare_split_features(self, split, type, save_file):
        interval = ''
        if type == 'hog':
            interval = self.hog_interval
        elif type == 'hof':
            interval = self.hof_interval
        else:
            interval = self.mbh_interval

        features_dict = {}
        for video, label in split:
            # print('Loading IDT feature of {} with label {} ... '.format(video, label), end='')
            # t0 = clock()
            feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME.format(self.params.trajectory_length, self.params.temporal_stride))
            # feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME_TEMP)
            features_dict[video] = (feature[:, interval], label)
            # print('%.4f seconds' % (clock() - t0))

        if save_file:
            self.save_features_to_pickle(save_file + FEATURE_DICT_POSTFIX, features_dict)

        features_cat = []
        for video, f in features_dict.items():
            features_cat.append(f[0])

        if save_file:
            self.save_features_to_pickle(save_file + FEATURE_CAT_POSTFIX, np.vstack(features_cat))

