from data.data_helper import DataHelper
from pipeline.encoding import generate_fisher_vector
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from pipeline.dimensionality_reduction import pca
from pipeline.clustering import generate_gmms
from time import time


import numpy as np
import pickle
import os
import pipeline.constants as const


class Extractor:

    train_data = {
        const.TRAJ: [],
        const.HOG: [],
        const.HOF: [],
        const.MBH: []
    }

    test_data = {
        const.TRAJ: [],
        const.HOG: [],
        const.HOF: [],
        const.MBH: []
    }

    def __init__(self, params, data_helper:DataHelper):
        self.data_helper = data_helper
        self.params = params

    @staticmethod
    def get_fisher_vectors(data, data_helper, pca, gmm, is_normalized):
        data_fv = []
        labels = []

        for k, v in data_helper:
            pca_d = pca.transform(data[k][0])
            fv_d = generate_fisher_vector(pca_d, gmm['means'], gmm['covars'], gmm['priors'], _normalized=is_normalized)
            data_fv.append(np.transpose(fv_d))
            labels.append(v)

        return np.asarray(data_fv), labels

    @staticmethod
    def load_txt(filename):
        return np.array(np.loadtxt(filename))

    @staticmethod
    def load_mat(filename):
        return loadmat(filename)

    @staticmethod
    def save_features_to_pickle(filename, file):
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

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
        skeleton = self.load_mat(os.path.join(self.params.data_path, video, 'skeleton.mat'))
        # skeleton = self.load_mat(os.path.join('/raid/users/oozdemir/data/BosphorusSign/ToyDataset', video, 'skeleton.mat'))
        sampled_features = self.check_hand_radius(features, skeleton['skeleton'], self.params.hand_radius)
        return np.asarray(sampled_features)

    def check_hand_radius(self, features, skeleton, hand_radius):
        tl = self.params.trajectory_length

        local_features_temp = []
        for f in features:
            frame_idx = int(f[0])
            trajStartX = f[1]
            trajStartY = f[2]
            trajEndX = f[2*tl-1]
            trajEndY = f[2*tl]

            # # head
            # headStartX = skeleton['Head'][0, 0][frame_idx - 15 + 1, 7] / 3
            # headStartY = skeleton['Head'][0, 0][frame_idx - 15 + 1, 8] / 3
            # headEndX = skeleton['Head'][0, 0][frame_idx, 7] / 3
            # headEndY = skeleton['Head'][0, 0][frame_idx, 8] / 3

            # left hand
            handLeftStartX = skeleton['HandLeft'][0, 0][frame_idx - 15 + 1, 7] / 3
            handLeftStartY = skeleton['HandLeft'][0, 0][frame_idx - 15 + 1, 8] / 3
            handLeftEndX = skeleton['HandLeft'][0, 0][frame_idx, 7] / 3
            handLeftEndY = skeleton['HandLeft'][0, 0][frame_idx, 8] / 3

            # right hand
            handRightStartX = skeleton['HandRight'][0, 0][frame_idx - 15 + 1, 7] / 3
            handRightStartY = skeleton['HandRight'][0, 0][frame_idx - 15 + 1, 8] / 3
            handRightEndX = skeleton['HandRight'][0, 0][frame_idx, 7] / 3
            handRightEndY = skeleton['HandRight'][0, 0][frame_idx, 8] / 3

            distLeftStart = euclidean((trajStartX, trajStartY), (handLeftStartX, handLeftStartY))
            distLeftEnd = euclidean((trajEndX, trajEndY), (handLeftEndX, handLeftEndY))
            distRightStart = euclidean((trajStartX, trajStartY), (handRightStartX, handRightStartY))
            distRightEnd = euclidean((trajEndX, trajEndY), (handRightEndX, handRightEndY))

            # distHeadStart = euclidean((trajStartX, trajStartY), (headStartX, headStartY))
            # distHeadEnd = euclidean((trajEndX, trajEndY), (headEndX, headEndY))
            if distLeftStart <= hand_radius and distLeftEnd <= hand_radius:
                local_features_temp.append(f)
            elif distRightStart <= hand_radius and distRightEnd <= hand_radius:
                local_features_temp.append(f)
            # elif distHeadStart <= hand_radius and distHeadEnd <= hand_radius:
            #     local_features_temp.append(f)

        return local_features_temp

    def clear_train_features(self):
        self.train_data = {
            const.TRAJ: [],
            const.HOG: [],
            const.HOF: [],
            const.MBH: []
        }

    def clear_test_features(self):
        self.test_data = {
            const.TRAJ: [],
            const.HOG: [],
            const.HOF: [],
            const.MBH: []
        }

    def prepare_train_features(self, type):
        pass

    def prepare_test_features(self, type):
        pass

    def set_features(self, return_dict=False):
        pass

    def prepare_data(self, type):
        pass

    def load_fisher_data_if_exists(self, fisher_path, type):
        if os.path.isfile(fisher_path + '.pickle'):
            print('Fisher data is exists for {}, loading... '.format(type), end='')
            t0 = time()
            fisher_data = self.load_features_from_pickle(fisher_path)
            train_fisher = fisher_data['data']['train_' + type + '_fisher']
            test_fisher = fisher_data['data']['test_' + type + '_fisher']
            train_labels = fisher_data['labels']['train_labels']
            test_labels = fisher_data['labels']['test_labels']
            print('%.4f seconds' % (time() - t0), flush=True)
            return train_fisher, test_fisher, train_labels, test_labels

        return None

    def process_model_if_not_exists(self, type):
        model_path = os.path.join(self.data_helper.save_path, 'model_' + type)
        if os.path.isfile(model_path + '.pickle'):
            print('PCA and GMM models are exists for {}, loading... '.format(type), end='')
            t0 = time()
            model = self.load_features_from_pickle(model_path)
            model_pca = model['pca']
            model_gmm = model['gmm']
        else:
            print('PCA and GMM models are not exists for {}, generating... '.format(type), end='')
            t0 = time()
            model_pca = pca(self.train_data[type])
            train_pca = model_pca.transform(self.train_data[type])
            model_gmm = generate_gmms(train_pca, _clusters=self.params.num_clusters)
            self.save_features_to_pickle(self.data_helper.save_path + os.path.sep + 'model_' + type, {'pca': model_pca, 'gmm': model_gmm})

        print('%.4f seconds' % (time() - t0), flush=True)
        return model_pca, model_gmm