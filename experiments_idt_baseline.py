from pipeline.features.idt_extractor import IdtExtractor
from pipeline.dimensionality_reduction import pca
from pipeline.clustering import generate_gmms
from pipeline.model_classifier import classify_svm

from data.data_helper import ToyDataHelper
from parameter_parser import ParameterParser

from os.path import sep

import numpy as np


def prepare_data(extrc, data_helper, type):
    extrc.read_training_features(type)
    extrc.prepare_train_features(type, return_dict=False)

    model_pca = pca(extrc.train_data[type])
    train_pca = model_pca.transform(extrc.train_data[type])
    model_gmm = generate_gmms(train_pca, _clusters=params.num_clusters)
    extrc.save_features_to_pickle(data_helper.save_path + sep + 'model_' + type, {'pca': model_pca, 'gmm': model_gmm})

    train_fisher = extrc.get_fisher_vectors(extrc.train_data[type], model_pca, model_gmm)
    extrc.clear_train_features()

    extrc.prepare_test_features(type)
    test_fisher = extrc.get_fisher_vectors(extrc.train_data[type], model_pca, model_gmm)
    extrc.clear_test_features()

    data = {'train_' + type + '_fisher': train_fisher, 'test_' + type + '_fisher': test_fisher}
    labels = {'train_labels': data_helper.train_labels, 'test_labels': data_helper.test_labels}
    extrc.save_features_to_pickle(data_helper.save_path + sep + 'fisher_data_' + type, {'data': data, 'labels': labels})
    return train_fisher, test_fisher

if __name__ == '__main__':
    """ Read Parameters """
    params = ParameterParser('idt').params

    """ Read Dataset Parameters """
    data_helper = ToyDataHelper(params, exp_type='idt')

    """ Initialize Extractor """
    extractor = IdtExtractor(params, data_helper)

    """ Extract fisher vectors """
    train_hog_fisher, test_hog_fisher = prepare_data(extractor, data_helper, 'hog')
    train_hof_fisher, test_hof_fisher = prepare_data(extractor, data_helper, 'hof')
    train_mbh_fisher, test_mbh_fisher = prepare_data(extractor, data_helper, 'mbh')

    ## Classifiy ##
    classify_svm(train_hog_fisher, data_helper.train_labels,
                 test_hog_fisher, data_helper.test_labels,
                 save_file=data_helper.save_path + sep + 'results_hog')
    classify_svm(train_hof_fisher, data_helper.train_labels,
                 test_hof_fisher, data_helper.test_labels,
                 save_file=data_helper.save_path + sep + 'results_hof')
    classify_svm(train_mbh_fisher, data_helper.train_labels,
                 test_mbh_fisher, data_helper.test_labels,
                 save_file=data_helper.save_path + sep  + 'results_mbh')

    classify_svm(np.hstack((train_hog_fisher, train_hof_fisher)), data_helper.train_labels,
                 np.hstack((test_hog_fisher, test_hof_fisher)), data_helper.test_labels,
                 save_file=data_helper.save_path + sep + 'results_hog_hof')
    classify_svm(np.hstack((train_hog_fisher, train_mbh_fisher)), data_helper.train_labels,
                 np.hstack((test_hog_fisher, test_mbh_fisher)), data_helper.test_labels,
                 save_file=data_helper.save_path + sep + 'results_hog_mbh')
    classify_svm(np.hstack((train_hof_fisher, train_mbh_fisher)), data_helper.train_labels,
                 np.hstack((test_hof_fisher, test_mbh_fisher)), data_helper.test_labels,
                 save_file=data_helper.save_path + sep + 'results_hof_mbh')

    classify_svm(np.hstack((train_hog_fisher, train_hof_fisher, train_mbh_fisher)), data_helper.train_labels,
                 np.hstack((test_hog_fisher, test_hof_fisher, test_mbh_fisher)), data_helper.test_labels,
                 save_file=data_helper.save_path + sep + 'results_hog_hof_mbh')


