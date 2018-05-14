from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import os
import sys
import h5py
from time import time


def classify_svm(train_data, train_labels, test_data, test_labels, save_file, is_logging=True):
    if is_logging:
        old_stdout = sys.stdout
        log_file = open(save_file + '.log', 'w')
        sys.stdout = log_file

    print("====================================================")
    print("SVM Classification")
    print("====================================================")
    print("====================================================")

    t0 = time()
    Cs = np.power(2, np.linspace(-3, 9, num=7))

    # svc = GridSearchCV(LinearSVC(class_weight='balanced', verbose=False), cv=5, param_grid=dict(C=Cs), n_jobs=-1)
    svc = GridSearchCV(SVC(kernel='linear', class_weight='balanced', probability=True, verbose=False), cv=5, param_grid=dict(C=Cs), n_jobs=-1)
    svc.fit(train_data, np.transpose(train_labels).ravel())
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best parameters: ")
    print(svc.best_params_)
    print()
    print("Best estimator: ")
    print(svc.best_estimator_)
    print()
    print("Best score: ")
    print(svc.best_score_)
    print()

    print("Started SVM prediction on test set ")
    t0 = time()
    predicted_labels = svc.predict(test_data)
    predict_scores = svc.predict_proba(test_data)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Accuracy Score: %f" % (100 * accuracy_score(np.transpose(test_labels), predicted_labels)))
    print()
    print("Top k labels: ")
    for idx in np.arange(0, len(predict_scores)):
        top_k_label = np.argsort(predict_scores[idx])[::1][-5:]
        print(
            "True label: %d, Predicted top 5 labels: %s" % (test_labels[idx], ','.join(str(e + 1) for e in top_k_label)))
    print()
    print(classification_report(test_labels, predicted_labels))
    print()
    print(confusion_matrix(test_labels, predicted_labels, labels=range(1, 10)))
    print()

    if is_logging:
        sys.stdout = old_stdout
        log_file.close()