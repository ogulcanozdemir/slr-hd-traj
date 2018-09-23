import argparse


class ParameterParser:

    params = None

    def __init__(self, exp_type):
        if exp_type == 'idt':
            self.params = self.parse_arguments_idt()
        elif exp_type == 'idt_hand':
            self.params = self.parse_arguments_idt_hand()
        elif exp_type == 'fhd':
            self.params = self.parse_arguments_fhd()
        elif exp_type == 'fhd_tf':
            self.params = self.parse_arguments_fhd_tf()

    @staticmethod
    def parse_arguments_idt():
        parser = argparse.ArgumentParser(description='IDT training')
        parser.add_argument('-dp', action='store', dest='data_path', help='data input path')
        parser.add_argument('-trp', action='store', dest='training_split_path', help='training_split_path')
        parser.add_argument('-tsp', action='store', dest='test_split_path', help='test split path')
        parser.add_argument('-cip', action='store', dest='class_ind_path', help='class indices path')
        parser.add_argument('-ep', action='store', dest='experiment_path', help='experiment path')
        parser.add_argument('-kf', action='store', type=int, dest='key_frames', help='has key frames??')
        parser.add_argument('-tl', action='store', type=int, dest='trajectory_length', help='trajectory length')
        parser.add_argument('-nt', action='store', type=int, dest='temporal_stride', help='temporal stride')
        parser.add_argument('-k', action='store', type=int, dest='num_clusters', help='number of clusters for gmm')
        parser.add_argument('--normalized', action='store_true', dest='normalized', help='are fisher vectors normalized??')

        params = parser.parse_args()
        for arg in vars(params):
            print('\'' + arg + '\':', getattr(params, arg))

        return params

    @staticmethod
    def parse_arguments_idt_hand():
        parser = argparse.ArgumentParser(description='IDT hand training')
        parser.add_argument('-dp', action='store', dest='data_path', help='data input path')
        parser.add_argument('-trp', action='store', dest='training_split_path', help='training_split_path')
        parser.add_argument('-tsp', action='store', dest='test_split_path', help='test split path')
        parser.add_argument('-cip', action='store', dest='class_ind_path', help='class indices path')
        parser.add_argument('-ep', action='store', dest='experiment_path', help='experiment path')
        parser.add_argument('-kf', action='store', type=int, dest='key_frames', help='has key frames??')
        parser.add_argument('-tl', action='store', type=int, dest='trajectory_length', help='trajectory length')
        parser.add_argument('-nt', action='store', type=int, dest='temporal_stride', help='temporal stride')
        parser.add_argument('-k', action='store', type=int, dest='num_clusters', help='number of clusters for gmm')
        parser.add_argument('-hr', action='store', type=int, dest='hand_radius', help='radius parameters for hand')
        parser.add_argument('--normalized', action='store_true', dest='normalized', help='are fisher vectors normalized??')

        params = parser.parse_args()
        for arg in vars(params):
            print('\'' + arg + '\':', getattr(params, arg))

        return params

    @staticmethod
    def parse_arguments_fhd():
        parser = argparse.ArgumentParser(description='IDT hand training')
        parser.add_argument('-dp', action='store', dest='data_path', help='data input path')
        parser.add_argument('-descp', action='store', dest='desc_path', help='descriptor output path')
        parser.add_argument('-trp', action='store', dest='training_split_path', help='training_split_path')
        parser.add_argument('-tsp', action='store', dest='test_split_path', help='test split path')
        parser.add_argument('-cip', action='store', dest='class_ind_path', help='class indices path')
        parser.add_argument('-ep', action='store', dest='experiment_path', help='experiment path')
        parser.add_argument('-kf', action='store', type=int, dest='key_frames', help='has key frames??')
        # parser.add_argument('-nt', action='store', type=int, dest='temporal_stride', help='temporal stride')
        parser.add_argument('-k', action='store', type=int, dest='num_clusters', help='number of clusters for gmm')
        parser.add_argument('--normalized', action='store_true', dest='normalized', help='are fisher vectors normalized??')
        parser.add_argument('-crop-size', action='store', type=int, dest='crop_size', help='crop size')
        parser.add_argument('-nbins', action='store', type=int, dest='nbins', help='number of orientations')
        parser.add_argument('-ncell', action='store', type=int, dest='ncell', help='cell size')
        parser.add_argument('-nblock', action='store', type=int, dest='nblock', help='block size')
        parser.add_argument('-resize', action='store_true', dest='resize', help='are fisher vectors normalized??')

        params = parser.parse_args()
        for arg in vars(params):
            print('\'' + arg + '\':', getattr(params, arg))

        return params

    @staticmethod
    def parse_arguments_fhd_tf():
        parser = argparse.ArgumentParser(description='LSTM training')
        parser.add_argument('-dp', action='store', dest='data_path', help='data input path')
        parser.add_argument('-descp', action='store', dest='desc_path', help='descriptor output path')
        parser.add_argument('-trp', action='store', dest='training_split_path', help='training_split_path')
        parser.add_argument('-tsp', action='store', dest='test_split_path', help='test split path')
        parser.add_argument('-cip', action='store', dest='class_ind_path', help='class indices path')
        parser.add_argument('-ep', action='store', dest='experiment_path', help='experiment path')
        parser.add_argument('-kf', action='store', type=int, dest='key_frames', help='has key frames??')
        parser.add_argument('-lr', action='store', type=float, dest='learning_rate', help='learning rate')
        parser.add_argument('-e', action='store', type=int, dest='epochs', help='number of epochs')
        parser.add_argument('-b', action='store', type=int, dest='batch_size', help='batch size')
        parser.add_argument('-nhlstm', action='store', type=int, dest='nh_lstm', help='number of hidden units for lstm')
        parser.add_argument('-dlstm', action='store', type=float, dest='d_lstm', help='dropout for lstm')
        parser.add_argument('-nhfc', action='store', type=int, dest='nh_fc', help='number of hidden units for fc')
        parser.add_argument('-dfc', action='store', type=float, dest='d_fc', help='dropout for fc')

        parser.add_argument('-crop-size', action='store', type=int, dest='crop_size', help='crop size')
        parser.add_argument('-nbins', action='store', type=int, dest='nbins', help='number of orientations')
        parser.add_argument('-ncell', action='store', type=int, dest='ncell', help='cell size')
        parser.add_argument('-nblock', action='store', type=int, dest='nblock', help='block size')

        params = parser.parse_args()
        for arg in vars(params):
            print('\'' + arg + '\':', getattr(params, arg))

        return params
