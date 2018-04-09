from time import clock
from os.path import sep

import numpy as np
import pickle

FEATURE_FIXED_NAME = 'color.features'


class IdtExtractor:

    train_trajectories_hog = None
    train_trajectories_hof = None
    train_trajectories_mbh = None

    test_trajectories_hog = None
    test_trajectories_hof = None
    test_trajectories_mbh = None

    def __init__(self, params):




def load_features(file, return_dict=False):
    if return_dict:
        features = load_features_from_pickle(file + '_dict')
    else:
        features = load_features_from_pickle(file + '_cat')

    return features


def prepare_features(path, split, save_file):
    features_dict = {}
    for video, label in split.items():
        print('Loading IDT feature of {} with label {} ... '.format(video, label), end='')
        t0 = clock()
        feature = load(path + sep + video + sep + FEATURE_FIXED_NAME)
        features_dict[video] = feature
        print('%.4f seconds' % (clock() - t0))

    if save_file:
        save_features_to_pickle(save_file + '_dict', features_dict)

    features_cat = []
    for video, f in features_dict.items():
        features_cat.append(f)

    if save_file:
        save_features_to_pickle(save_file + '_cat', np.vstack(features_cat))


def load(filename):
    return np.array(np.loadtxt(filename))


def save_features_to_pickle(filename, file):
    with open(filename + '.pickle', 'wb') as fd:
        pickle.dump(file, fd)


def load_features_from_pickle(file):
    with open(file + '.pickle', 'rb') as fd:
        features = pickle.load(fd)

    return features