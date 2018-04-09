from scipy.io import loadmat

import os
import json
import numpy as np


class DataSample:

    hog = None
    hof = None
    mbh = None


class DataHelper:

    training_split = None
    test_split = None

    train_labels = None
    test_labels = None

    def __init__(self, params:ParameterHelper):
        data_params = params.data_params
        experiment_params = params.experiments_params

        self.data_path = data_params['path']
        self.experiment_id = experiment_params['id']
        self.experiment_path = experiment_params['path']
        self.split_path = data_params['split_path']
        self.save_path = os.path.join(self.experiment_path, 'experiment_'+self.experiment_id)
        self.descriptors_path = os.path.join(self.save_path, 'descriptors')
        self.feature_path = os.path.join(self.save_path, 'cnn_features')

    def prepare_splits(self):
        self.training_split = []
        self.test_split = []
        self.train_labels = []
        self.test_labels = []

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

    def __init__(self, params:ParameterHelper):
        DataHelper.__init__(self, params)

    def prepare_splits(self):
        DataHelper.prepare_splits(self)

        training_split_mat = loadmat(os.path.join(self.split_path, 'train_split_py.mat'))
        training_split_mat = training_split_mat['trainSplit'][0]
        for spl in training_split_mat:
            sample_class = spl['class'][0]
            sample_video = spl['video'][0]
            self.training_split.append((sample_video, sample_class))
            self.train_labels.append(sample_class)

        test_split_mat = loadmat(os.path.join(self.split_path, 'test_split_py.mat'))
        test_split_mat = test_split_mat['testSplit'][0]
        for spl in test_split_mat:
            sample_class = spl['class'][0]
            sample_video = spl['video'][0]
            self.test_split.append((sample_video, sample_class))
            self.test_labels.append(sample_class)