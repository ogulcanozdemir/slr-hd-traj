from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier

# from imblearn.ensemble import BalancedBaggingClassifier
# from imblearn.metrics import classification_report_imbalanced
import numpy as np
import sys
from time import time


def classify_svm(train_data, train_labels, test_data, test_labels, save_file, is_logging=True, _njobs=1):
    if is_logging:
        old_stdout = sys.stdout
        log_file = open(save_file + '.log', 'w')
        sys.stdout = log_file

    print("====================================================", flush=True)
    print("SVM Classification", flush=True)
    print("====================================================", flush=True)
    print("====================================================", flush=True)

    t0 = time()
    # Cs = np.power(2, np.linspace(-3, 5, num=5))

    params = {
        'base_estimator__C': np.power(2, np.linspace(-3, 5, num=5)),
        'n_estimators': [10],
        # 'max_samples': [0.3],
        # 'random_state': [0]
    }

    # svc = GridSearchCV(BaggingClassifier(SVC(kernel='linear', probability=True, verbose=False, cache_size=2000), n_jobs=_njobs), param_grid=params, pre_dispatch='1*n_jobs')
    # svc = GridSearchCV(BalancedBaggingClassifier(SVC(kernel='linear', probability=True, verbose=False, cache_size=2000), n_jobs=_njobs), param_grid=params, pre_dispatch='1*n_jobs')
    # svc = GridSearchCV(LinearSVC(class_weight='balanced', verbose=False), cv=5, param_grid=dict(C=Cs), n_jobs=-1)
    # svc = GridSearchCV(SVC(kernel='linear', probability=True, verbose=False, cache_size=2000), param_grid=dict(C=Cs), pre_dispatch='1*n_jobs', n_jobs=_njobs)
    Cs = np.power(2, np.linspace(-3, 9, num=7))
    svc = GridSearchCV(LinearSVC(class_weight='balanced', verbose=False), cv=5, param_grid=dict(C=Cs), pre_dispatch='1*n_jobs', n_jobs=_njobs)
    svc.fit(train_data, np.transpose(train_labels).ravel())
    print("done in %0.3fs" % (time() - t0), flush=True)
    print(flush=True)
    print("Best parameters: ", flush=True)
    print(svc.best_params_, flush=True)
    print(flush=True)
    print("Best estimator: ")
    print(svc.best_estimator_, flush=True)
    print(flush=True)
    print("Best score: ", flush=True)
    print(svc.best_score_, flush=True)
    print(flush=True)
    print("Started SVM prediction on test set ", flush=True)
    t0 = time()
    predicted_labels = svc.predict(test_data)
    # predict_scores = svc.predict_proba(test_data)
    print("done in %0.3fs" % (time() - t0), flush=True)
    print(flush=True)
    print("Accuracy Score: %f" % (100 * accuracy_score(np.transpose(test_labels), predicted_labels)), flush=True)
    print(flush=True)
    # print("Top k labels: ", flush=True)
    # for idx in np.arange(0, len(predict_scores)):
    #     top_k_label = np.argsort(predict_scores[idx])[::1][-5:]
    #     print("True label: %d, Predicted top 5 labels: %s" % (test_labels[idx], ','.join(str(e + 1) for e in top_k_label)), flush=True)
    # print(flush=True)
    print(classification_report(test_labels, predicted_labels), flush=True)
    # print(classification_report_imbalanced(test_labels, predicted_labels), flush=True)
    print(flush=True)
    cm = confusion_matrix(test_labels, predicted_labels, labels=range(1, 153))
    np.save(save_file + '.npy', cm)
    print(cm, flush=True)
    print(flush=True)

    # with open('svc_model_imbalanced_default.pickle', 'wb') as fp:
    #     pickle.dump(svc, fp)

    if is_logging:
        sys.stdout = old_stdout
        log_file.close()