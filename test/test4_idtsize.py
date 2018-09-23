import pickle
import json
import numpy as np
import os


data_path = '/raid/users/oozdemir/test/ToyDataset'

n_training_samples = 305

siz = 0
_classdir = os.listdir(data_path)
for c in _classdir:
    _userdir = os.listdir(os.path.join(data_path, c))
    for u in _userdir:
        print(c, u)
        if os.path.isfile(os.path.join(data_path, c, u, 'color.features')):
            feature = np.array(np.loadtxt(os.path.join(data_path, c, u, 'color.features')))
            siz += feature.shape[0]

avg_features = siz / n_training_samples
print(avg_features)


# with open(hog_path, 'rb') as fhog:
#     features_hog = pickle.load(fhog)
#
# n_features_hog, hog_dim =  features_hog.shape
#
# with open(hof_path, 'rb') as fhof:
#     features_hof = pickle.load(fhof)
#
# n_features_hof, hof_dim =  features_hof.shape
#
# with open(mbh_path, 'rb') as fmbh:
#     features_mbh = pickle.load(fmbh)
#
# n_features_mbh, mbh_dim =  features_mbh.shape
#
# mx_features = np.max((n_features_hog, n_features_hof, n_features_mbh))
# avg_features = mx_features / n_training_samples
# sum_dim = hog_dim + hof_dim + mbh_dim
#
# print('number of features per sign: {}, number of features: {}, feature size: {}'.format(avg_features, mx_features, sum_dim))