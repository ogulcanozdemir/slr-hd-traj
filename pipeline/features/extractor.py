from data.data_helper import DataHelper
from pipeline.encoding import generate_fisher_vector
from scipy.io import loadmat
from scipy.spatial.distance import euclidean

import numpy as np
import pickle
import os

class Extractor:

    train_data = {
        'hog': [],
        'hof': [],
        'mbh': []
    }

    test_data = {
        'hog': [],
        'hof': [],
        'mbh': []
    }

    def __init__(self, params, data_helper:DataHelper):
        self.data_helper = data_helper
        self.params = params

    @staticmethod
    def get_fisher_vectors(data, pca, gmm):
        data_fv = []

        for k, v in data.items():
            pca_d = pca.transform(v[0])
            fv_d = generate_fisher_vector(pca_d, gmm['means'], gmm['covars'], gmm['priors'])
            data_fv.append(np.transpose(fv_d))

        return np.asarray(data_fv)

    @staticmethod
    def load_txt(filename):
        return np.array(np.loadtxt(filename))

    @staticmethod
    def load_mat(filename):
        return loadmat(filename)

    @staticmethod
    def save_features_to_pickle(filename, file):
        with open(filename + '.pickle', 'wb') as fd:
            pickle.dump(file, fd, protocol=4)

    @staticmethod
    def load_features_from_pickle(file):
        with open(file + '.pickle', 'rb') as fd:
            features = pickle.load(fd)

        return features

    def load_features(self, file, return_dict=False):
        if return_dict:
            features = self.load_features_from_pickle(file + '_dict')
        else:
            features = self.load_features_from_pickle(file + '_cat')

        return features

    def get_hand_trajectories_from_video(self, features, video):
        skeleton = self.load_mat(os.path.join(self.params.skeleton_path, video, 'skeleton.mat'))
        sampled_features = self.check_hand_radius(features, skeleton['skeleton'], self.params.hand_radius)
        return sampled_features

    def check_hand_radius(self, features, skeleton, hand_radius):
        tl = self.params.trajectory_length

        local_features_temp = []
        for f in features:
            frame_idx = features[0]
            trajStartX = features[1]
            trajStartY = features[2]
            trajEndX = features[2*tl-1]
            trajEndY = features[2*tl]

            # left hand
            handLeftStartX = skeleton['HandLeft'](frame_idx - 15 + 1, 7) / 3
            handLeftStartY = skeleton['HandLeft'](frame_idx - 15 + 1, 8) / 3
            handLeftEndX = skeleton['HandLeft'](frame_idx, 7) / 3
            handLeftEndY = skeleton['HandLeft'](frame_idx, 8) / 3

            # right hand
            handRightStartX = skeleton['HandRight'](frame_idx - 15 + 1, 7) / 3
            handRightStartY = skeleton['HandRight'](frame_idx - 15 + 1, 8) / 3
            handRightEndX = skeleton['HandRight'](frame_idx, 7) / 3
            handRightEndY = skeleton['HandRight'](frame_idx, 8) / 3

            distLeftStart = euclidean((trajStartX, trajStartY), (handLeftStartX, handLeftStartY))
            distLeftEnd = euclidean((trajEndX, trajEndY), (handLeftEndX, handLeftEndY))
            distRightStart = euclidean((trajStartX, trajStartY), (handRightStartX, handRightStartY))
            distRightEnd = euclidean((trajEndX, trajEndY), (handRightEndX, handRightEndY))
            if distLeftStart <= hand_radius and distLeftEnd <= hand_radius:
                local_features_temp.append(f)
            elif distRightStart <= hand_radius and distRightEnd <= hand_radius:
                local_features_temp.append(f)

        return local_features_temp

    def clear_train_features(self):
        self.train_data = {
            'hog': [],
            'hof': [],
            'mbh': []
        }

    def clear_test_features(self):
        self.test_data = {
            'hog': [],
            'hof': [],
            'mbh': []
        }

    def prepare_train_features(self, type):
        pass

    def prepare_test_features(self, type):
        pass

    def set_features(self, return_dict=False):
        pass
