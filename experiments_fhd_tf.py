from pipeline.features.fhd_extractor import FhdExtractor
from pipeline.model_builder import ModelBuilder
from data.data_helper import ToyDataHelper
from parameter_parser import ParameterParser

from os.path import sep

import pipeline.constants as const
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM


def generator(features, labels, batch_size=1):
    while True:
        index = np.random.choice(len(features),1)
        yield features[index], labels[index]


if __name__ == '__main__':
    exp_type = 'fhd_tf'
    params = ParameterParser(exp_type).params
    extract = False

    """ Read Dataset Parameters """
    data_helper = ToyDataHelper(params, exp_type=exp_type)

    """ Initialize Extractor """
    extractor = FhdExtractor(params, data_helper)

    train_data, test_data, train_labels, test_labels = extractor.prepare_data_for_tf(seq_len=140)

    model = Sequential()
    model.add(LSTM(units=250, return_sequences=True, input_shape=(None, 264)))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(generator(train_data, train_labels), epochs=5, validation_data=(test_data, test_labels))

    # model = ModelBuilder(lr=params.learning_rate,
    #                      epochs=params.epochs,
    #                      batch_size=params.batch_size,
    #                      num_classes=data_helper.num_classes,
    #                      seq_len=140,
    #                      input_dim=264,
    #                      m_lstm='1xlstm',
    #                      is_bidirectional=True,
    #                      nh_lstm=params.nh_lstm,
    #                      d_lstm=params.d_lstm,
    #                      nh_fc=2048,
    #                      d_fc=0.25)
    #
    # model.build()
    # model.train((train_data, train_labels), (test_data, test_labels),
    #             save_file=data_helper.save_path + sep + 'results_' + const.HOG + '_' + const.HOF + '_' + const.MBH,
    #             is_logging=False)



    print('Finished...')