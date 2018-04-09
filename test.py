from data.split import load_split
from pipeline.features.idt_extractor import load_features, prepare_features
from pipeline.dimensionality_reduction import pca
from pipeline.clustering import generate_gmms

from os.path import sep

import os


if __name__ == '__main__':

    data_folder = os.getcwd() + sep + 'data'
    toydata_path = data_folder + sep + 'ToyDataset'
    split_path = data_folder + sep + 'splits'
    save_path = data_folder + sep + 'features' + sep + 'idt_features'

    # read splits
    training_split = load_split(split_path + sep + 'train-copy.txt', key_frames=False)
    test_split = load_split(split_path + sep + 'test-copy.txt', key_frames=False)

    # prepare features (concat etc..)
    # prepare_features(toydata_path, training_split, save_file=save_path + '_training')
    # prepare_features(toydata_path, test_split, save_file=save_path + '_test')

    # load concatenated features or dictionary
    training_features_cat = load_features(save_path + '_training', return_dict=False)

    pca_tra = pca(training_features_cat)
    model_tra = generate_gmms(pca_tra.transform(training_features_cat))

    test_features = load_features(save_path + '_test', return_dict=False)
