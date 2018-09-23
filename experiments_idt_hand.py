from pipeline.features.idt_extractor import IdtExtractor
from pipeline.model_classifier import classify_svm

from data.data_helper import ToyDataHelper
from parameter_parser import ParameterParser

from os.path import sep, isfile, join

import numpy as np
import pipeline.constants as const


if __name__ == '__main__':
    """ Read Parameters """
    exp_type = 'idt_hand'
    params = ParameterParser(exp_type).params

    """ Read Dataset Parameters """
    data_helper = ToyDataHelper(params, exp_type=exp_type)

    """ Initialize Extractor """
    extractor = IdtExtractor(params, data_helper)

    """ Extract fisher vectors """
    train_traj_fisher, test_traj_fisher, train_labels, test_labels = extractor.prepare_data(const.TRAJ)
    train_hog_fisher, test_hog_fisher, _, _ = extractor.prepare_data(const.HOG)
    train_hof_fisher, test_hof_fisher, _, _ = extractor.prepare_data(const.HOF)
    train_mbh_fisher, test_mbh_fisher, _, _ = extractor.prepare_data(const.MBH)

    _n_jobs = 20

    results_prefix = 'results_'

    # ## Classifiy ##
    # classify_svm(train_traj_fisher, train_labels,
    #              test_traj_fisher, test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.TRAJ,
    #              _njobs=_n_jobs)
    # classify_svm(train_hog_fisher, train_labels,
    #              test_hog_fisher, test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.HOG,
    #              _njobs=_n_jobs)
    # classify_svm(train_hof_fisher, train_labels,
    #              test_hof_fisher, test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.HOF,
    #              _njobs=_n_jobs)
    # classify_svm(train_mbh_fisher, train_labels,
    #              test_mbh_fisher, test_labels,
    #              save_file=data_helper.save_path + sep  + results_prefix + const.MBH,
    #              _njobs=_n_jobs)
    #
    # classify_svm(np.hstack((train_hog_fisher, train_hof_fisher)), train_labels,
    #              np.hstack((test_hog_fisher, test_hof_fisher)), test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.HOG + '_' + const.HOF,
    #              _njobs=_n_jobs)
    # classify_svm(np.hstack((train_hog_fisher, train_mbh_fisher)), train_labels,
    #              np.hstack((test_hog_fisher, test_mbh_fisher)), test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.HOG + '_' + const.MBH,
    #              _njobs=_n_jobs)
    # classify_svm(np.hstack((train_hof_fisher, train_mbh_fisher)), train_labels,
    #              np.hstack((test_hof_fisher, test_mbh_fisher)), test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.HOF + '_' + const.MBH,
    #              _njobs=_n_jobs)

    classify_svm(np.hstack((train_hog_fisher, train_hof_fisher, train_mbh_fisher)), train_labels,
                 np.hstack((test_hog_fisher, test_hof_fisher, test_mbh_fisher)), test_labels,
                 save_file=data_helper.save_path + sep + results_prefix + const.HOG + '_' + const.HOF + '_' + const.MBH,
                 _njobs=_n_jobs)
    # classify_svm(np.hstack((train_traj_fisher, train_hog_fisher, train_hof_fisher, train_mbh_fisher)), train_labels,
    #              np.hstack((test_traj_fisher, test_hog_fisher, test_hof_fisher, test_mbh_fisher)), test_labels,
    #              save_file=data_helper.save_path + sep + results_prefix + const.TRAJ + '_' + const.HOG + '_' + const.HOF + '_' + const.MBH,
    #              _njobs=_n_jobs)

