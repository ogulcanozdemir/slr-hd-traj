from pipeline.features.extractor import Extractor

from time import clock
from os.path import sep

import numpy as np

FEATURE_FIXED_NAME = 'color.features'
FEATURE_FOLDER_NAME = 'idt-features'
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
        self.feature_root = self.data_helper.feature_path + sep + FEATURE_FOLDER_NAME

    def set_features(self, return_dict=False):
        train_features = self.load_features(self.feature_root + sep + FEATURE_TRAIN_PREFIX, return_dict=return_dict)
        if return_dict: # if return_dict is True => training mode, if its False preprocessing mode
            for spl in self.data_helper.training_split:
                self.train_hog.append((spl[0], spl[1], train_features[spl[0]][0][:, self.hog_interval]))
                self.train_hof.append((spl[0], spl[1], train_features[spl[0]][0][:, self.hof_interval]))
                self.train_mbh.append((spl[0], spl[1], train_features[spl[0]][0][:, self.mbh_interval]))

            test_features = self.load_features(self.feature_root + sep + FEATURE_TEST_PREFIX, return_dict=return_dict)
            for spl in self.data_helper.test_split:
                self.test_hog.append((spl[0], spl[1], test_features[spl[0]][0][:, self.hog_interval]))
                self.test_hof.append((spl[0], spl[1], test_features[spl[0]][0][:, self.hof_interval]))
                self.test_mbh.append((spl[0], spl[1], test_features[spl[0]][0][:, self.mbh_interval]))
        else:
            self.train_hog = train_features[:, self.hog_interval]
            self.train_hof = train_features[:, self.hof_interval]
            self.train_mbh = train_features[:, self.mbh_interval]

    def prepare_features(self):
        self.prepare_split_features(self.data_helper.training_split,
                                    save_file=self.feature_root + sep + FEATURE_TRAIN_PREFIX)
        self.prepare_split_features(self.data_helper.test_split,
                                    save_file=self.feature_root + sep + FEATURE_TEST_PREFIX)

    def prepare_split_features(self, split, save_file):
        features_dict = {}
        for video, label in split:
            print('Loading IDT feature of {} with label {} ... '.format(video, label), end='')
            t0 = clock()
            feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME)
            features_dict[video] = (feature, label)
            print('%.4f seconds' % (clock() - t0))

        if save_file:
            self.save_features_to_pickle(save_file + FEATURE_DICT_POSTFIX, features_dict)

        features_cat = []
        for video, f in features_dict.items():
            features_cat.append(f[0])

        if save_file:
            self.save_features_to_pickle(save_file + FEATURE_CAT_POSTFIX, np.vstack(features_cat))

