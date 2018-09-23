from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
import seaborn as sns
import pandas as pd

import pickle
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if __name__ == '__main__':
    data_path = '/raid/users/oozdemir/code/tm-shd-slr/experiments_2/idt_hand_general/idt_hand_hr35_l20_t3/k64/'

    fv_data_file = data_path + 'fisher_data_traj.pickle'
    data_file = data_path + 'results_hog_hof_mbh.npy'
    cf = np.load(data_file)

    cf = cf[0:154, 0:154]
    print(cf)


    with open(fv_data_file, 'rb') as fp:
        fisher_data = pickle.load(fp)

    train_fisher = np.array(fisher_data['data']['train_traj_fisher'])
    test_fisher = np.array(fisher_data['data']['test_traj_fisher'])
    train_labels = np.array(fisher_data['labels']['train_labels'])
    test_labels = np.array(fisher_data['labels']['test_labels'])
    #
    # with open('C:\\Users\\ogulc\\Desktop\\tm-shd-slr\\svc_model_ms03.pickle', 'rb') as fp:
    #     svc = pickle.load(fp)
    #
    # bcls = svc.best_estimator_
    # bcls_estimators = bcls.estimators_
    # bcls_estimators_samples = bcls.estimators_samples_

    classes = np.unique(test_labels)

    # fig = plt.figure(figsize=(4, 5))
    # outer = gridspec.GridSpec(2, 5, wspace=.2, hspace=.3)

    # for est_idx in range(1):
    #     inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[est_idx], wspace=.2, hspace=.5)

        # j = 0
        # sample_arr = bcls_estimators_samples[est_idx]
        # samples_freq = train_labels[sample_arr]
        # hist, _ = np.histogram(samples_freq)
        # ax = plt.Subplot(fig, inner[j])
        # ax.bar(classes, hist)
        # ax.set_xticks(classes)
        # title = 'SVM #'+str(est_idx+1)+', num_samples '+str(np.sum(hist))
        # ax.set_title(title)
        # fig.add_subplot(ax)

        # j = 1
        # est = bcls_estimators[est_idx]
        # predicted_labels = classes.take(est.predict(test_fisher), axis=0)
        # cm = confusion_matrix(test_labels, predicted_labels, labels=range(1, 11))
        # ax = plt.Subplot(fig, inner[0])

    annot = np.empty_like(cf).astype(str)
    nrows, ncols = cf.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cf[i, j]
            annot[i, j] = '%d' % (c)

    df_cm = pd.DataFrame(cf, index=classes, columns=classes)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    svm = sns.heatmap(df_cm, annot=annot, fmt='', cbar=False, cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True))

    figure = svm.get_figure()
    figure.savefig('svm_conf.png', dpi=400)



    # cmd = pd.DataFrame(cf, index=classes, columns=classes)
    # sns_plot = sns.heatmap(cmd, annot=annot, fmt='', cbar=False, cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True))
        # fig.add_subplot(ax)

    # sns_plot.savefig("sample-test.png", dpi=(100))
    # fig.show()
    print(132)
    # predict probabilities
    # predicted_proba = bcls.predict_proba(test_fisher)
    # # majority voting
    # predicted_labels = classes.take((np.argmax(predicted_proba, axis=1)), axis=0)
    #
    # print(100 * accuracy_score(np.transpose(test_labels), predicted_labels))
    #
    # # predict for each estimator
    # for est_idx in range(0, len(bcls_estimators)):
    #     print('Estimator #%d' % est_idx)
    #     predicted_labels = bcls_estimators[est_idx].predict(test_fisher)
    #     print(100 * accuracy_score(np.transpose(test_labels), predicted_labels))