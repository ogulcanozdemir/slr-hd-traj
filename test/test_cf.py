import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import seaborn as sns
import pandas as pd


class ConfusionMatrixGrapher:
    def _init_(self):
        pass

    @staticmethod
    def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
        """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

        Arguments
        ---------
        confusion_matrix: numpy.ndarray
            The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
            Similarly constructed ndarrays can also be used.
        class_names: list
            An ordered list of class names, in the order they index the given confusion matrix.
        figsize: tuple
            A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: int
            Font size for axes labels. Defaults to 14.

        Returns
        -------
        matplotlib.figure.Figure
            The resulting confusion matrix figure
        """

        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        boundaries = [0.0, 0.5, 1.0]

        hex_colors = sns.light_palette('navy', n_colors=len(boundaries)*2, as_cmap=False).as_hex()
        hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]

        colors = list(zip(boundaries, hex_colors))

        custom_color_map = LinearSegmentedColormap.from_list(
            name='custom_navy',
            colors=['blue', 'cyan', 'green', 'yellow','red'],
        )

        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=False, cbar=True, xticklabels=15, yticklabels=15, cmap=custom_color_map)
            # heatmap = sns.heatmap(df_cm, annot=False, cbar=False, xticklabels=15, yticklabels=15, cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True))
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        # heatmap.yaxis.set_ticklabels(10, rotation=0, ha='right', fontsize=fontsize)
        # heatmap.xaxis.set_ticklabels(10, rotation=0, ha='right', fontsize=fontsize)
        plt.yticks(rotation=0)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{0:.3f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{0}".format(int(cm[i, j])),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        plt.savefig('fhd_cp80_nc40_nbl2_nbin8_k64.png', dpi=400)


if __name__ == '__main__':
    data_path = '/raid/users/oozdemir/code/tm-shd-slr/experiments_2/idt_hand_general_variance_inconsistent/idt_hand_hr45_l20_t3/k64/'
    fv_data_file = data_path + 'fisher_data_hog.pickle'
    data_file = data_path + 'results_hog_hof_mbh.npy'
    cf = np.load(data_file)

    # cf = cf[0:152, 0:152]
    np.savetxt(data_path+'/cm.txt', cf.astype(int), fmt='%i')

    print(cf)

    with open(fv_data_file, 'rb') as fp:
        fisher_data = pickle.load(fp)

    train_fisher = np.array(fisher_data['data']['train_hog_fisher'])
    test_fisher = np.array(fisher_data['data']['test_hog_fisher'])
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

    # ConfusionMatrixGrapher.plot_confusion_matrix(cf, classes)
    ConfusionMatrixGrapher.print_confusion_matrix(cf, classes).savefig(data_path+'/cm.png', dpi=400)