import argparse


class ParameterParser:

    params = None

    def __init__(self, exp_type):
        if exp_type == 'idt':
            self.params = self.parse_arguments_idt()

    @staticmethod
    def parse_arguments_idt():
        parser = argparse.ArgumentParser(description='IDT training')
        parser.add_argument('-dp', action='store', dest='data_path', help='data input path')
        parser.add_argument('-trp', action='store', dest='training_split_path', help='training_split_path')
        parser.add_argument('-tsp', action='store', dest='test_split_path', help='test split path')
        parser.add_argument('-ep', action='store', dest='experiment_path', help='experiment path')
        parser.add_argument('-kf', action='store', type=int, dest='key_frames', help='has key frames??')
        parser.add_argument('-tl', action='store', type=int, dest='trajectory_length', help='trajectory length')
        parser.add_argument('-nt', action='store', type=int, dest='temporal_stride', help='temporal stride')
        parser.add_argument('-k', action='store', type=int, dest='num_clusters', help='number of clusters for gmm')

        params = parser.parse_args()
        for arg in vars(params):
            print('\'' + arg + '\':', getattr(params, arg))

        return params

    @staticmethod
    def parse_arguments(self):
        # parse arguments
        parser = argparse.ArgumentParser(description='LSTM training')
        parser.add_argument('-i', action='store', dest='input_path', help='input feature path')
        parser.add_argument('-t', action='store', dest='training_split_path', help='training split (txt)')
        parser.add_argument('-v', action='store', dest='validation_split_path', help='validation split (txt)')
        parser.add_argument('-lr', action='store', type=float, dest='learning_rate', help='learning rate')
        parser.add_argument('-e', action='store', type=int, dest='num_epoch', help='number of epochs')
        parser.add_argument('-b', action='store', type=int, dest='batch_size', help='batch size')
        parser.add_argument('-nc', action='store', type=int, dest='num_classes', help='number of classes')
        parser.add_argument('-sl', action='store', type=int, dest='seq_len', help='sequence length')
        parser.add_argument('-dim', action='store', type=int, dest='input_dim', help='input dimensions')
        parser.add_argument('-mlstm', action='store', dest='m_lstm', help='model lstm e.g.(2xLSTM, 2xBasicLSTM, 2xGRU)')
        parser.add_argument('-bilstm', action='store', type=int, dest='is_bidirectional', help='is bidirectional')
        parser.add_argument('-nhlstm', action='store', type=int, dest='nh_lstm', help='number of hidden units for lstm')
        parser.add_argument('-dlstm', action='store', type=float, dest='d_lstm', help='dropout for lstm')
        parser.add_argument('-nhfc', action='store', type=int, dest='nh_fc', help='number of hidden units for fc')
        parser.add_argument('-dfc', action='store', type=float, dest='d_fc', help='dropout for lstm')

        params = parser.parse_args()
        for arg in vars(params):
            print('\'' + arg + '\':', getattr(params, arg))

        return params
