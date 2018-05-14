from data.data_helper import DataHelper
from pipeline.encoding import generate_fisher_vector

import numpy as np
import pickle


class Extractor:

    train_hog = []
    train_hof = []
    train_mbh = []

    test_hog = []
    test_hof = []
    test_mbh = []

    def __init__(self, params, data_helper:DataHelper):
        self.data_helper = data_helper
        self.params = params

    @staticmethod
    def get_fisher_vectors(data, pca, gmm):
        data_fv = []

        for d in data:
            pca_d = pca.transform(d[2])
            fv_d = generate_fisher_vector(pca_d, gmm['means'], gmm['covars'], gmm['priors'])
            data_fv.append(np.transpose(fv_d))

        return np.asarray(data_fv)

    @staticmethod
    def load_txt(filename):
        return np.array(np.loadtxt(filename))

    @staticmethod
    def save_features_to_pickle(filename, file):
        with open(filename + '.pickle', 'wb') as fd:
            pickle.dump(file, fd)

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

    def clear_train_features(self):
        self.train_hog = []
        self.train_hof = []
        self.train_mbh = []

    def clear_test_features(self):
        self.test_hog = []
        self.test_hof = []
        self.test_mbh = []

    def prepare_train_features(self):
        pass

    def prepare_test_features(self):
        pass

    def set_features(self, return_dict=False):
        pass
