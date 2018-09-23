from pipeline.features.extractor import Extractor
from pipeline.dimensionality_reduction import pca
from pipeline.clustering import generate_gmms

from os.path import sep

import numpy as np
import os
import pipeline.constants as const


FEATURE_FIXED_NAME = 'color_scaled_L{}_T{}.features'
FEATURE_FIXED_NAME_TEMP = 'color.features'
FEATURE_TRAIN_PREFIX = 'train_idt'
FEATURE_TEST_PREFIX = 'test_idt'
FEATURE_DICT_POSTFIX = '_dict'
FEATURE_CAT_POSTFIX = '_cat'


class IdtExtractor(Extractor):

    hog_size = 96
    hof_size = 108
    mbh_size = 192

    def __init__(self, params, data_helper):
        Extractor.__init__(self, params, data_helper)
        self.intervals = {
            const.TRAJ: lambda vl: range(vl - (self.hog_size + self.hof_size + self.mbh_size + 2 * params.trajectory_length), vl - (self.hog_size + self.hof_size + self.mbh_size)),
            const.HOG: lambda vl: range(vl - (self.hog_size + self.hof_size + self.mbh_size), vl - (self.hof_size + self.mbh_size)),
            const.HOF: lambda vl: range(vl - (self.hof_size + self.mbh_size), vl - (self.mbh_size)),
            const.MBH: lambda vl: range(vl - (self.mbh_size), vl)
        }

    def read_training_features(self, type):
        self.prepare_split_features(self.data_helper.training_split, save_file=self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX)

    def read_test_features(self, type):
        self.prepare_split_features(self.data_helper.test_split, save_file=self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX)

    def prepare_train_features(self, type, return_dict=False):
        self.train_data[type] = self.load_features(self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type, return_dict=return_dict)

    def prepare_test_features(self, type):
        self.test_data[type] = self.load_features(self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX + '_' + type, return_dict=True)

    def prepare_split_features(self, split, save_file):
        if os.path.isfile(save_file + '_' + const.HOG + FEATURE_DICT_POSTFIX + '.pickle') and os.path.isfile(save_file + '_' + const.HOG + FEATURE_CAT_POSTFIX + '.pickle'):
            print(save_file + FEATURE_DICT_POSTFIX + ' files are exists, skipping ...')
            return

        features_traj = {}
        features_hog = {}
        features_hof = {}
        features_mbh = {}
        for video, label in split:
            # feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME.format(self.params.trajectory_length, self.params.temporal_stride))
            feature = self.load_txt(self.data_helper.data_path + sep + video + sep + FEATURE_FIXED_NAME_TEMP)
            if self.data_helper.feature_path.find('hr') > -1:
                feature = self.get_hand_trajectories_from_video(feature, video)

            features_traj[video] = (feature[:, self.intervals[const.TRAJ](feature.shape[1])], label)
            features_hog[video] = (feature[:, self.intervals[const.HOG](feature.shape[1])], label)
            features_hof[video] = (feature[:, self.intervals[const.HOF](feature.shape[1])], label)
            features_mbh[video] = (feature[:, self.intervals[const.MBH](feature.shape[1])], label)

        if save_file:
            self.save_features_to_pickle(save_file + '_' + const.TRAJ + FEATURE_DICT_POSTFIX, features_traj)
            self.save_features_to_pickle(save_file + '_' + const.HOG + FEATURE_DICT_POSTFIX, features_hog)
            self.save_features_to_pickle(save_file + '_' + const.HOF + FEATURE_DICT_POSTFIX, features_hof)
            self.save_features_to_pickle(save_file + '_' + const.MBH + FEATURE_DICT_POSTFIX, features_mbh)

        features_traj_cat = []
        for video, f in features_traj.items():
            features_traj_cat.append(f[0])

        features_hog_cat = []
        for video, f in features_hog.items():
            features_hog_cat.append(f[0])

        features_hof_cat = []
        for video, f in features_hof.items():
            features_hof_cat.append(f[0])

        features_mbh_cat = []
        for video, f in features_mbh.items():
            features_mbh_cat.append(f[0])

        if save_file:
            self.save_features_to_pickle(save_file + '_' + const.TRAJ + FEATURE_CAT_POSTFIX, np.vstack(features_traj_cat))
            self.save_features_to_pickle(save_file + '_' + const.HOG + FEATURE_CAT_POSTFIX, np.vstack(features_hog_cat))
            self.save_features_to_pickle(save_file + '_' + const.HOF + FEATURE_CAT_POSTFIX, np.vstack(features_hof_cat))
            self.save_features_to_pickle(save_file + '_' + const.MBH + FEATURE_CAT_POSTFIX, np.vstack(features_mbh_cat))

    def prepare_data(self, type):
        fisher_path = os.path.join(self.data_helper.save_path,'fisher_data_' + type)
        if os.path.isfile(fisher_path + '.pickle'):
            fisher_data = self.load_features_from_pickle(fisher_path)
            train_fisher = fisher_data['data']['train_' + type + '_fisher']
            test_fisher = fisher_data['data']['test_' + type + '_fisher']
            train_labels = fisher_data['labels']['train_labels']
            test_labels = fisher_data['labels']['test_labels']
            return train_fisher, test_fisher, train_labels, test_labels

        self.read_training_features(type)
        self.prepare_train_features(type, return_dict=False)

        model_path = os.path.join(self.data_helper.save_path, 'model_' + type)
        if os.path.isfile(model_path + '.pickle'):
            model = self.load_features_from_pickle(model_path)
            model_pca = model['pca']
            model_gmm = model['gmm']
        else:
            model_pca = pca(self.train_data[type])
            train_pca = model_pca.transform(self.train_data[type])
            model_gmm = generate_gmms(train_pca, _clusters=self.params.num_clusters)
            self.save_features_to_pickle(self.data_helper.save_path + sep + 'model_' + type, {'pca': model_pca, 'gmm': model_gmm})

        self.clear_train_features()
        self.prepare_train_features(type, return_dict=True)
        train_fisher, train_labels = self.get_fisher_vectors(self.train_data[type], self.data_helper.training_split, model_pca, model_gmm, is_normalized=self.params.normalized)
        self.clear_train_features()

        self.read_test_features(type)
        self.prepare_test_features(type)
        test_fisher, test_labels = self.get_fisher_vectors(self.test_data[type], self.data_helper.test_split, model_pca, model_gmm, is_normalized=self.params.normalized)
        self.clear_test_features()

        data = {'train_' + type + '_fisher': train_fisher, 'test_' + type + '_fisher': test_fisher}
        labels = {'train_labels': train_labels, 'test_labels': test_labels}
        self.save_features_to_pickle(fisher_path, {'data': data, 'labels': labels})
        return train_fisher, test_fisher, train_labels, test_labels


