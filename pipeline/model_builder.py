from tensorflow.contrib import rnn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing.label import label_binarize
from pipeline.batch_loader import BatchLoader

import tensorflow as tf
import numpy as np
import sys


class ModelBuilder:

    display_step = 1

    def __init__(self,
                 lr,
                 epochs,
                 batch_size,
                 num_classes,
                 seq_len,
                 input_dim,
                 m_lstm,
                 nh_lstm,
                 is_bidirectional=False,
                 d_lstm=None,
                 nh_fc=None,
                 d_fc=None):
        self.learning_rate = lr
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.is_bidirectional = is_bidirectional

        self.num_lstm, self.lstm_type = m_lstm.split('x')
        self.num_lstm = int(self.num_lstm)

        self.nh_lstm = nh_lstm
        self.d_lstm = d_lstm if d_lstm != 0 else None
        self.nh_fc = nh_fc if nh_fc != 0 else None
        self.d_fc = d_fc if d_fc != 0 else None

        self.lstm_weights = {}
        self.lstm_biases = {}

    def initialize_lstm_weights(self):
        outs = self.nh_fc if self.nh_fc is not None else self.num_classes

        self.lstm_weights = {
            'lstm': tf.Variable(tf.random_normal([(2 if self.is_bidirectional else 1) * self.nh_lstm, outs])),
        }

        self.lstm_biases = {
            'lstm': tf.Variable(tf.random_normal([outs])),
        }

    def build(self):
        self.X = tf.placeholder(tf.float32, [None, self.seq_len, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])

        # initialize lstm layers
        self.initialize_lstm_weights()
        out = self.get_lstm_model()

        # initialize fully connected layers
        if self.nh_fc is not None:
            out = tf.layers.dense(out, units=self.nh_fc, activation=tf.nn.relu)
            if self.d_fc is not None:
                out = tf.layers.dropout(inputs=out, rate=self.d_fc, training=True)

        # initialize prediction layer
        pred_out = tf.layers.dense(out, units=self.num_classes)
        self.prediction = tf.nn.softmax(pred_out)

        # define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_out, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, train, test, save_file=None, is_logging=True):
        if is_logging:
            old_stdout = sys.stdout
            log_file = open(save_file + '.log', 'w')
            sys.stdout = log_file

        # initialize batch loader
        batch_loader = BatchLoader(train[0], train[1], self.seq_len)

        # # prepare test data
        reshaped_test = test[0].reshape((test[0].shape[0], self.seq_len, self.input_dim))
        lb = LabelBinarizer()
        lb.fit(batch_loader.get_classes())
        binarized_test_labels = lb.fit_transform(test[1])

        # initialize the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # run the initializer
            sess.run(init)

            # for e in range(1, self.num_epochs+1):
            #     # iteration_count = int(np.ceil(train[0].shape[0] / self.batch_size))
            #     # for idx in range(0, iteration_count):
            #     batch_x, batch_y = batch_loader.next_batch(self.batch_size)
            #     batch_x = np.expand_dims(batch_x, axis=1)
            #     # batch_x = batch_x.reshape((self.batch_size, self.seq_len, self.input_dim))
            #
            #     sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
            #     if e % self.display_step == 0 or e == 1:
            #         # Calculate batch loss and accuracy
            #         loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
            #         print("Minibatch Step " + str(e) + ", Loss= {:.4f}".format(loss) + ", Training Accuracy= {:.3f}".format(acc))
            #
            #     if e % 100 == 0:
            #         predictions = []
            #         for smpl_idx in range(0, test[0].shape[0]):
            #             acc = []
            #             for desc in test[0][smpl_idx]:
            #                 acc.append(sess.run(self.prediction, feed_dict={self.X: desc.reshape((1, 1, self.input_dim)), self.Y: binarized_test_labels[smpl_idx, :].reshape((1, self.num_classes))}))
            #             sum_acc = np.sum(acc, axis=0)
            #             predictions.append(np.argmax(sum_acc))
            #
            #         # binarized_predictions = lb.fit_transform(predictions)
            #         print("Epoch #" + str(e) + ", Test Accuracy:", (100 * np.sum(test[1] == predictions)) / test[1].shape[0])
            #
            #     # if e % 100 == 0:
            #     #     # evaluate model every 100 iterations
            #     #     print("Epoch #" + str(e) + ", Test Accuracy:",
            #     #           sess.run(self.accuracy, feed_dict={self.X: reshaped_test, self.Y: binarized_test_labels}), flush=True)

            for step in range(1, self.num_epochs+1):
                batch_x, batch_y = batch_loader.next_batch(self.batch_size)
                batch_x = batch_x.reshape((self.batch_size, self.seq_len, self.input_dim))

                sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
                if step % self.display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

                if step % 100 == 0:
                    # evaluate model every 100 iterations
                    print("Testing Accuracy:", sess.run(self.accuracy, feed_dict={self.X: reshaped_test, self.Y: binarized_test_labels}))

            print("Optimization Finished!")

        if is_logging:
            sys.stdout = old_stdout
            log_file.close()

    def get_lstm_cell(self):
        cell = None
        if self.lstm_type.lower() == 'basiclstm':
            cell = rnn.BasicLSTMCell(self.nh_lstm, forget_bias=1.0, state_is_tuple=True)
        elif self.lstm_type.lower() == 'lstm':
            cell = rnn.LSTMCell(self.nh_lstm, forget_bias=1.0, state_is_tuple=True)
        else: # GRU
            cell = rnn.GRUCell(self.nh_lstm)

        if self.d_lstm is not None:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.d_lstm)

        return cell

    def get_lstm_model(self):
        if self.is_bidirectional:
            if self.num_lstm is not 1:
                lstm_fw_cells = [self.get_lstm_cell() for _ in range(self.num_lstm)]
                lstm_fw_cells = rnn.MultiRNNCell(lstm_fw_cells)
                lstm_bw_cells = [self.get_lstm_cell() for _ in range(self.num_lstm)]
                lstm_bw_cells = rnn.MultiRNNCell(lstm_bw_cells)
            else:
                lstm_fw_cells = self.get_lstm_cell()
                lstm_bw_cells = self.get_lstm_cell()

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells, self.X, dtype=tf.float32, time_major=False)
            outputs = tf.concat(outputs, 2)
        else:
            if self.num_lstm is not 1:
                lstm_cells = [self.get_lstm_cell() for _ in range(self.num_lstm)]
                lstm_cells = rnn.MultiRNNCell(lstm_cells)
            else:
                lstm_cells = self.get_lstm_cell()

            outputs, states = tf.nn.dynamic_rnn(lstm_cells, inputs=self.X, dtype=tf.float32, time_major=False)

        h = tf.transpose(outputs, [1, 0, 2])
        return tf.nn.xw_plus_b(h[-1], weights=self.lstm_weights['lstm'], biases=self.lstm_biases['lstm'])