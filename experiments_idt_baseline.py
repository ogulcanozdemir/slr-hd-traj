from pipeline.features.idt_extractor import IdtExtractor
from pipeline.dimensionality_reduction import pca
from pipeline.clustering import generate_gmms
from pipeline.model_classifier import classify_svm

from data.data_helper import ToyDataHelper
from parameter_parser import ParameterParser

from os.path import sep

import numpy as np


if __name__ == '__main__':
    params = ParameterParser('idt').params
    data_helper = ToyDataHelper(params, exp_type='idt')
    extractor = IdtExtractor(params, data_helper)

    ## HOG ##
    extractor.read_features('hog')
    extractor.prepare_train_features('hog', return_dict=False)

    pca_hog = pca(extractor.train_hog)
    train_hog_pca = pca_hog.transform(extractor.train_hog)
    gmm_hog = generate_gmms(train_hog_pca, _clusters=params.num_clusters)
    extractor.save_features_to_pickle(data_helper.save_path + sep + 'model_hog', {'pca': pca_hog, 'gmm': gmm_hog})

    extractor.clear_train_features()
    extractor.prepare_train_features('hog', return_dict=True)
    extractor.prepare_test_features('hog')
    train_hog_fisher = extractor.get_fisher_vectors(extractor.train_hog, pca_hog, gmm_hog)
    test_hog_fisher = extractor.get_fisher_vectors(extractor.test_hog, pca_hog, gmm_hog)

    data = {
        'train_hog_fisher': train_hog_fisher,
        'test_hog_fisher': test_hog_fisher,
    }

    labels = {
        'train_labels': data_helper.train_labels,
        'test_labels': data_helper.test_labels
    }
    extractor.save_features_to_pickle(data_helper.save_path + sep + 'fisher_data_hog', {'data': data, 'labels': labels})
    extractor.clear_train_features()
    extractor.clear_test_features()

    ## HOF ##
    extractor.read_features(type='hof')
    extractor.prepare_train_features('hof', return_dict=False)

    pca_hof = pca(extractor.train_hof)
    train_hof_pca = pca_hof.transform(extractor.train_hof)
    gmm_hof = generate_gmms(train_hof_pca, _clusters=params.num_clusters)
    extractor.save_features_to_pickle(data_helper.save_path + sep + 'model_hof', {'pca': pca_hof, 'gmm': gmm_hof})

    extractor.clear_train_features()
    extractor.prepare_train_features('hof', return_dict=True)
    extractor.prepare_test_features('hof')
    train_hof_fisher = extractor.get_fisher_vectors(extractor.train_hof, pca_hof, gmm_hof)
    test_hof_fisher = extractor.get_fisher_vectors(extractor.test_hof, pca_hof, gmm_hof)

    data = {
        'train_hof_fisher': train_hof_fisher,
        'test_hof_fisher': test_hof_fisher,
    }

    labels = {
        'train_labels': data_helper.train_labels,
        'test_labels': data_helper.test_labels
    }
    extractor.save_features_to_pickle(data_helper.save_path + sep + 'fisher_data_hof', {'data': data, 'labels': labels})
    extractor.clear_train_features()
    extractor.clear_test_features()

    ## MBH ##
    extractor.read_features(type='mbh')
    extractor.prepare_train_features('mbh', return_dict=False)

    pca_mbh = pca(extractor.train_mbh)
    train_mbh_pca = pca_mbh.transform(extractor.train_mbh)
    gmm_mbh = generate_gmms(train_mbh_pca, _clusters=params.num_clusters)
    extractor.save_features_to_pickle(data_helper.save_path + sep + 'model_mbh', {'pca': pca_mbh, 'gmm': gmm_mbh})

    extractor.clear_train_features()
    extractor.prepare_train_features('mbh', return_dict=True)
    extractor.prepare_test_features('mbh')
    train_mbh_fisher = extractor.get_fisher_vectors(extractor.train_mbh, pca_mbh, gmm_mbh)
    test_mbh_fisher = extractor.get_fisher_vectors(extractor.test_mbh, pca_mbh, gmm_mbh)

    data = {
        'train_mbh_fisher': train_mbh_fisher,
        'test_mbh_fisher': test_mbh_fisher,
    }

    labels = {
        'train_labels': data_helper.train_labels,
        'test_labels': data_helper.test_labels
    }
    extractor.save_features_to_pickle(data_helper.save_path + sep + 'fisher_data_mbh', {'data': data, 'labels': labels})
    extractor.clear_train_features()
    extractor.clear_test_features()

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


