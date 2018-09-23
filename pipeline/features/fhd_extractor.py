from pipeline.features.extractor import Extractor
from pipeline.utils.video_utils import read_video
from pipeline.utils.skeleton_utils import read_skeleton, find_hand_coordinates
from pipeline.utils.preprocessing_utils import get_hand_crop_from_video

from pipeline.features.descriptors.hog import extract_hog_from_video
from pipeline.features.descriptors.hof import calc_optical_flow_from_video, extract_hof_from_flow_seq
from pipeline.features.descriptors.mbh import extract_mbh_from_flow_seq
from tensorflow.python.keras.preprocessing import sequence

from joblib import Parallel, delayed
from time import clock, time
from os.path import sep
from scipy.misc import imsave
from sklearn.preprocessing import normalize

import os
import pickle
import numpy as np
import pipeline.constants as const
import tensorflow as tf


FEATURE_FIXED_NAME = 'fhd_crop{}_ncell{}_nblock{}_nbins{}'
FEATURE_TRAIN_PREFIX = 'train_fhd'
FEATURE_TEST_PREFIX = 'test_fhd'
FEATURE_DICT_POSTFIX = '_dict'
FEATURE_CAT_POSTFIX = '_cat'


class FhdExtractor(Extractor):

    def __init__(self, params, data_helper):
        Extractor.__init__(self, params, data_helper)

    def extract_descriptors(self, _n_jobs=None):
        _class_folders = os.listdir(self.data_helper.data_path)
        for _class in _class_folders:
            _video_folders = os.listdir(os.path.join(self.data_helper.data_path, _class))
            if _n_jobs is not None:
                Parallel(n_jobs=_n_jobs, pre_dispatch='1*n_jobs')(delayed(self.extract_descriptors_from_video)(_class, _video) for _video in _video_folders)
            else:
                for _video in _video_folders:
                    self.extract_descriptors_from_video(clazz=_class, video=_video)

    def extract_descriptors_from_video(self, clazz=None, video=None):
        print('Extracting descriptors from class: %s, video: %s... ' % (clazz, video), end='')
        t0 = time()
        video_path = os.path.join(os.path.join(self.data_helper.data_path, clazz, video))

        # video preprocessing
        frames = self.get_frames_from_video(video_path)
        l_hand, r_hand = self.get_hand_coordinates_from_skeleton(video_path, resize=self.params.resize)
        flow = self.get_optical_flow_from_video_frames(frames)

        # prepare hand crops
        l_hand_cropped_frames = self.get_hand_crops_from_frames(frames, l_hand)
        r_hand_cropped_frames = self.get_hand_crops_from_frames(frames, r_hand)
        l_hand_cropped_flow = self.get_hand_crops_from_frames(flow, l_hand, _is_flow=True)
        r_hand_cropped_flow = self.get_hand_crops_from_frames(flow, r_hand, _is_flow=True)

        # HOG, HOF and MBH
        hand_hog = self.get_hog_features((l_hand_cropped_frames, r_hand_cropped_frames))
        hand_hof = self.get_hof_features((l_hand_cropped_flow, r_hand_cropped_flow))
        hand_mbh = self.get_mbh_features((l_hand_cropped_flow, r_hand_cropped_flow))

        # collect descriptors
        desc = {
            'hog': np.asarray(hand_hog),
            'hof': np.asarray(hand_hof),
            'mbh': np.asarray(hand_mbh)
        }

        # save descriptors
        save_path = os.path.join(self.params.desc_path,
                                 clazz,
                                 video,
                                 'fhd_crop{}_ncell{}_nblock{}_nbins{}'.format(self.params.crop_size, self.params.ncell,
                                                                              self.params.nblock, self.params.nbins))
        self.save_features_to_pickle(save_path ,desc)
        print('%.4f seconds' % (time() - t0), flush=True)

    @staticmethod
    def get_frames_from_video(video_path):
        return read_video(video_path)

    @staticmethod
    def get_hand_coordinates_from_skeleton(video_path, resize):
        skeleton = read_skeleton(video_path)
        return find_hand_coordinates(skeleton, scaled=False)

    @staticmethod
    def get_optical_flow_from_video_frames(frames):
        return calc_optical_flow_from_video(frames)

    def get_hand_crops_from_frames(self, frames, skeleton, _is_flow=False):
        crop_frames = get_hand_crop_from_video(frames, skeleton, crop_size=self.params.crop_size, is_flow=_is_flow)
        return crop_frames

    def get_hog_features(self, frames):
        l_features = extract_hog_from_video(frames[0], num_orientations=self.params.nbins,
                                            cell_size=self.params.ncell, block_size=self.params.nblock, resize=self.params.resize)
        r_features = extract_hog_from_video(frames[1], num_orientations=self.params.nbins,
                                            cell_size=self.params.ncell, block_size=self.params.nblock, resize=self.params.resize)
        # return np.hstack((l_features, r_features))
        return np.hstack((r_features, l_features))

    def get_hof_features(self, flow):
        l_features = extract_hof_from_flow_seq(flow[0], num_orientations=self.params.nbins + 1, # add 1 more orientations [no motion]
                                               cell_size=self.params.ncell, block_size=self.params.nblock, resize=self.params.resize)
        r_features = extract_hof_from_flow_seq(flow[1], num_orientations=self.params.nbins + 1, # add 1 more orientations [no motion]
                                               cell_size=self.params.ncell, block_size=self.params.nblock, resize=self.params.resize)
        return np.hstack((r_features, l_features))

    def get_mbh_features(self, flow):
        l_features = extract_mbh_from_flow_seq(flow[0], num_orientations=self.params.nbins,
                                               cell_size=self.params.ncell, block_size=self.params.nblock, resize=self.params.resize)
        r_features = extract_mbh_from_flow_seq(flow[1], num_orientations=self.params.nbins,
                                               cell_size=self.params.ncell, block_size=self.params.nblock, resize=self.params.resize)
        return np.hstack((r_features, l_features))

    def prepare_split_features(self, split, save_file):
        if os.path.isfile(save_file + '_' + const.HOG + FEATURE_DICT_POSTFIX + '.pickle') and os.path.isfile(save_file + '_' + const.HOG + FEATURE_CAT_POSTFIX + '.pickle'):
            print(save_file + FEATURE_DICT_POSTFIX + ' files are exists, skipping ...')
            return

        features_hog = {}
        features_hof = {}
        features_mbh = {}
        for video, label in split:
            feature = self.load_features_from_pickle(self.data_helper.data_path + sep + video + sep + \
                                                     FEATURE_FIXED_NAME.format(self.params.crop_size, self.params.ncell,
                                                                               self.params.nblock, self.params.nbins))

            if hasattr(self.params, 'temporal_stride'):
                feature[const.HOG] = self.get_temporal_average(feature[const.HOG])
                feature[const.HOF] = self.get_temporal_average(feature[const.HOF])
                feature[const.MBH] = self.get_temporal_average(feature[const.MBH])

            features_hog[video] = (feature[const.HOG], label)
            features_hof[video] = (feature[const.HOF], label)
            features_mbh[video] = (feature[const.MBH], label)

        if save_file:
            self.save_features_to_pickle(save_file + '_' + const.HOG + FEATURE_DICT_POSTFIX, features_hog)
            self.save_features_to_pickle(save_file + '_' + const.HOF + FEATURE_DICT_POSTFIX, features_hof)
            self.save_features_to_pickle(save_file + '_' + const.MBH + FEATURE_DICT_POSTFIX, features_mbh)

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
            self.save_features_to_pickle(save_file + '_' + const.HOG + FEATURE_CAT_POSTFIX, np.vstack(features_hog_cat))
            self.save_features_to_pickle(save_file + '_' + const.HOF + FEATURE_CAT_POSTFIX, np.vstack(features_hof_cat))
            self.save_features_to_pickle(save_file + '_' + const.MBH + FEATURE_CAT_POSTFIX, np.vstack(features_mbh_cat))

    def read_training_features(self):
        self.prepare_split_features(self.data_helper.training_split, save_file=self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX)

    def read_test_features(self):
        self.prepare_split_features(self.data_helper.test_split, save_file=self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX)

    def prepare_train_features(self, type, return_dict=False):
        self.train_data[type] = self.load_features(self.data_helper.feature_path + sep + FEATURE_TRAIN_PREFIX + '_' + type, return_dict=return_dict)

    def prepare_test_features(self, type):
        self.test_data[type] = self.load_features(self.data_helper.feature_path + sep + FEATURE_TEST_PREFIX + '_' + type, return_dict=True)

    def prepare_data(self, type):
        fisher_path = os.path.join(self.data_helper.save_path,'fisher_data_' + type)
        fisher_data = self.load_fisher_data_if_exists(fisher_path, type)
        if fisher_data is not None:
            return fisher_data

        self.read_training_features()
        self.prepare_train_features(type, return_dict=False)

        model_pca, model_gmm = self.process_model_if_not_exists(type)

        self.clear_train_features()
        self.prepare_train_features(type, return_dict=True)
        print('Generating Training Fisher Vectors for {}... '.format(type), end='')
        t0 = time()
        train_fisher, train_labels = self.get_fisher_vectors(self.train_data[type], self.data_helper.training_split, model_pca, model_gmm, is_normalized=self.params.normalized)
        print('%.4f seconds' % (time() - t0), flush=True)
        self.clear_train_features()

        self.read_test_features()
        self.prepare_test_features(type)
        print('Generating Test Fisher Vectors for {}... '.format(type), end='')
        t0 = time()
        test_fisher, test_labels = self.get_fisher_vectors(self.test_data[type], self.data_helper.test_split, model_pca, model_gmm, is_normalized=self.params.normalized)
        print('%.4f seconds' % (time() - t0), flush=True)
        self.clear_test_features()

        data = {'train_' + type + '_fisher': train_fisher, 'test_' + type + '_fisher': test_fisher}
        labels = {'train_labels': train_labels, 'test_labels': test_labels}
        self.save_features_to_pickle(fisher_path, {'data': data, 'labels': labels})
        return train_fisher, test_fisher, train_labels, test_labels

    def prepare_data_for_tf(self, seq_len):
        # training data
        self.read_training_features()
        self.prepare_train_features(const.HOG, return_dict=True)
        self.prepare_train_features(const.HOF, return_dict=True)
        self.prepare_train_features(const.MBH, return_dict=True)

        train_data = []
        train_labels = []
        max_len = 0
        for sample, label in self.data_helper.training_split:
            hog = self.train_data[const.HOG][sample][0][1:] # hog features have one more row
            hof = self.train_data[const.HOF][sample][0]
            mbh = self.train_data[const.MBH][sample][0]
            max_len = hog.shape[0] if hog.shape[0] > max_len else max_len
            fused = np.hstack((normalize(hog, norm='l2'), normalize(hof, norm='l2'), normalize(mbh, norm='l2')))
            train_data.append(fused)
            train_labels.append(label)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        # tf.one_hot(np.asarray(train_labels), self.data_helper.num_classes, 1.0, 0.0)

        # test data
        self.read_test_features()
        self.prepare_test_features(const.HOG)
        self.prepare_test_features(const.HOF)
        self.prepare_test_features(const.MBH)

        test_data = []
        test_labels = []
        for sample, label in self.data_helper.test_split:
            hog = self.test_data[const.HOG][sample][0][1:] # hog features have one more row
            hof = self.test_data[const.HOF][sample][0]
            mbh = self.test_data[const.MBH][sample][0]
            max_len = hog.shape[0] if hog.shape[0] > max_len else max_len
            fused = np.hstack((normalize(hog, norm='l2'), normalize(hof, norm='l2'), normalize(mbh, norm='l2')))
            test_data.append(fused)
            test_labels.append(label)

        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)
        # tf.one_hot(np.asarray(test_labels), self.data_helper.num_classes, 1.0, 0.0)
        print('dataset max sequence length (max frame count): {}'.format(max_len))

        # new_d = []
        # for d in train_data:
        #     new_d.append(sequence.pad_sequences(np.transpose(d), maxlen=seq_len, padding='post', dtype='float64'))
        # train_data = np.asarray(new_d)
        #
        # new_d = []
        # for d in test_data:
        #     new_d.append(sequence.pad_sequences(np.transpose(d), maxlen=seq_len, padding='post', dtype='float64'))
        # test_data = np.asarray(new_d)

        return train_data, test_data, train_labels, test_labels

    def get_temporal_average(self, feature):
        nt = self.params.temporal_stride

        rpt = np.mod(feature.shape[0], nt)
        dup_rows_avg = np.mean(feature[-rpt:], axis=0)
        f = np.vstack((feature[0:-rpt], np.repeat([dup_rows_avg], repeats=nt, axis=0)))

        new_f = []
        indices = np.arange(0, f.shape[0]).reshape([int(f.shape[0] / nt), nt])
        [new_f.append(np.mean(f[idx, :], axis=0)) for idx in indices]

        return new_f


