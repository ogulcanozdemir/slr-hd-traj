import numpy as np
import pickle
import os

from tensorflow.examples.tutorials.mnist import input_data


from tensorflow.python.keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer


class BatchLoader:

    _index_in_epoch = 0
    _epochs_completed = 0
    _max_len = 0
    features = []
    labels = []
    key_frames = []

    def __init__(self, data, labels, sequence_length):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.features = np.asarray(data)
        self.labels = labels

        lb = LabelBinarizer()
        lb.fit(self.get_classes())
        self.binarized_labels = lb.fit_transform(labels)

        self._num_examples = self.features.shape[0]
        self._max_len = sequence_length

    def get_classes(self):
        return np.unique(self.labels)

    def num_classes(self):
        return len(self.get_classes())

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._binarized_labels = self.binarized_labels[perm0, :]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._binarized_labels[start:self._num_examples, :]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self.features[perm]
                self._binarized_labels = self.binarized_labels[perm, :]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            feature_new_part = self._features[start:end]
            labels_new_part = self._binarized_labels[start:end, :]
            return np.concatenate((features_rest_part, feature_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._binarized_labels[start:end, :]

    # def next_batch(self, batch_size, shuffle=True):
    #     start = self._index_in_epoch
    #     # Shuffle for the first epoch
    #     if self._epochs_completed == 0 and start == 0 and shuffle:
    #         perm0 = np.arange(self._num_examples)
    #         np.random.shuffle(perm0)
    #         self._features = self.features[perm0]
    #         self._binarized_labels = self.binarized_labels[perm0, :]
    #         self._binarized_labels = [np.repeat(self._binarized_labels[f, np.newaxis, :], self._features[f].shape[0], axis=0) for f in range(0, self._features.shape[0])]
    #         self._binarized_labels = np.concatenate(self._binarized_labels, axis=0)
    #         self._features = np.concatenate(self._features, axis=0)
    #         self._num_examples_expanded = self._features.shape[0]
    #     # Go to the next epoch
    #     if start + batch_size > self._num_examples_expanded:
    #         # Finished epoch
    #         self._epochs_completed += 1
    #         # Get the rest examples in this epoch
    #         rest_num_examples = self._num_examples_expanded - start
    #         features_rest_part = self._features[start:self._num_examples_expanded]
    #         labels_rest_part = self._binarized_labels[start:self._num_examples_expanded, :]
    #         # Shuffle the data
    #         if shuffle:
    #             perm = np.arange(self._num_examples)
    #             np.random.shuffle(perm)
    #             self._features = self.features[perm]
    #             self._binarized_labels = self.binarized_labels[perm, :]
    #             self._binarized_labels = [np.repeat(self._binarized_labels[f, np.newaxis, :], self._features[f].shape[0], axis=0) for f in range(0, self._features.shape[0])]
    #             self._binarized_labels = np.concatenate(self._binarized_labels, axis=0)
    #             self._features = np.concatenate(self._features, axis=0)
    #             self._num_examples_expanded = self._features.shape[0]
    #         # Start next epoch
    #         start = 0
    #         self._index_in_epoch = batch_size - rest_num_examples
    #         end = self._index_in_epoch
    #         feature_new_part = self._features[start:end]
    #         labels_new_part = self._binarized_labels[start:end, :]
    #         return np.concatenate((features_rest_part, feature_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
    #
    #     else:
    #         self._index_in_epoch += batch_size
    #         end = self._index_in_epoch
    #         return self._features[start:end], self._binarized_labels[start:end, :]