from pipeline.utils.video_utils import read_video
from pipeline.utils.skeleton_utils import read_skeleton, find_hand_coordinates
from pipeline.utils.preprocessing_utils import get_hand_crop_from_video, get_hand_crop_from_flow_seq

from pipeline.features.descriptors.hog import extract_hog_from_video
from pipeline.features.descriptors.hof import extract_hof_from_flow_seq, calc_optical_flow_from_video
from pipeline.features.descriptors.mbh import extract_mbh_from_flow_seq
from pipeline.encoding import generate_fisher_vector

from pipeline.features.extractor import Extractor

from parameter_helper import ParameterHelper
from time import clock
from data.data_helper import DataHelper, DataSample

import os
import numpy as np
import json


class DescExtractor(Extractor):

    descriptors_path = None

    cell_size = None
    block_size = None
    num_orientations = None
    normalize = None
    visualize = None
    dist2wrist = None
    crop_border = None

    train_descriptors_hog = None
    train_descriptors_hof = None
    train_descriptors_mbh = None

    test_descriptors_hog = None
    test_descriptors_hof = None
    test_descriptors_mbh = None

    def __init__(self, params: ParameterHelper, data_helper:DataHelper):
        desc_params = params.hyper_params['desc']
        crop_params = params.hyper_params['crop']

        self.cell_size = desc_params['pixels_per_cell']
        self.block_size = desc_params['cells_per_block']
        self.num_orientations = desc_params['nOrientations']
        self.normalize = desc_params['normalize']
        self.visualize = desc_params['visualize']
        self.ntCells = desc_params['ntCells']
        self.length = desc_params['length']
        self.dist2wrist = crop_params['dist2wrist']
        self.crop_border = crop_params['crop_border']
        self.target_size = crop_params['target_size']
        self.data_helper = data_helper
        self.descriptors_path = os.path.join(self.data_helper.save_path, 'descriptors')

    def prepare_folders(self):
        print('Creating folders to store descriptors... ', end='')
        t0 = clock()
        if not os.path.exists(self.descriptors_path):
            os.makedirs(self.descriptors_path)

        _class_folders = os.listdir(self.data_helper.data_path)
        for _class in _class_folders:
            _class_folder = os.path.join(self.descriptors_path, _class)
            if not os.path.exists(_class_folder):
                os.makedirs(_class_folder)

        print('%.4f seconds' % (clock() - t0))

    def extract_descriptors(self):
        _class_folders = os.listdir(self.data_helper.data_path)
        for _class in _class_folders:
            _video_folders = os.listdir(os.path.join(self.data_helper.data_path, _class))
            for _video in _video_folders:
                self.extract_descriptors_from_video(clazz=_class, video=_video)

    def extract_descriptors_from_video(self, clazz=None, video=None):
        print('Extracting descriptors from class: %s, video: %s... ' % (clazz, video), end='')
        t0 = clock()
        video_path = os.path.join(os.path.join(self.data_helper.data_path, clazz, video))

        # video preprocessing
        frames = self.get_frames_from_video(video_path, aligned=False)
        skeleton = self.get_skeleton_from_kinect(video_path)
        flows = self.get_optical_flow_from_video_frames(frames)

        # prepare hand crops
        l_hand, r_hand = self.__get_hand_coordinates_from_skeleton(skeleton, source='kinect', modality='rgb')
        l_hand_cropped_frames = self.__get_hand_crops_from_frames(frames, l_hand)
        r_hand_cropped_frames = self.__get_hand_crops_from_frames(frames, r_hand)
        l_hand_cropped_flows = self.__get_hand_crops_from_flows(flows, l_hand)
        r_hand_cropped_flows = self.__get_hand_crops_from_flows(flows, r_hand)

        # HOG
        l_hand_hog = self.__get_hog_features(l_hand_cropped_frames)
        r_hand_hog = self.__get_hog_features(r_hand_cropped_frames)

        # HOF
        l_hand_hof = self.__get_hof_features(l_hand_cropped_flows)
        r_hand_hof = self.__get_hof_features(r_hand_cropped_flows)

        # MBH
        l_hand_mbhx, l_hand_mbhy = self.__get_mbh_features(l_hand_cropped_flows)
        r_hand_mbhx, r_hand_mbhy = self.__get_mbh_features(r_hand_cropped_flows)

        # collect descriptors
        descriptors = {
            'hog_left': np.asarray(l_hand_hog).tolist(),
            'hog_right': np.asarray(r_hand_hog).tolist(),
            'hof_left': np.asarray(l_hand_hof).tolist(),
            'hof_right': np.asarray(r_hand_hof).tolist(),
            'mbhx_left': np.asarray(l_hand_mbhx).tolist(),
            'mbhy_left': np.asarray(l_hand_mbhy).tolist(),
            'mbhx_right': np.asarray(r_hand_mbhx).tolist(),
            'mbhy_right': np.asarray(r_hand_mbhy).tolist(),
        }

        # save descriptors
        self.save_descriptors(clazz, video, descriptors)

        print('%.4f seconds' % (clock() - t0))

    @staticmethod
    def get_frames_from_video(video_path, aligned=True):
        frames = np.asarray(read_video(video_path, aligned=aligned))
        return frames

    @staticmethod
    def get_optical_flow_from_video_frames(frames):
        flows = calc_optical_flow_from_video(frames)
        return flows

    @staticmethod
    def get_skeleton_from_openpose(video_path):
        skeleton = read_skeleton(video_path, source='openpose')
        return skeleton

    @staticmethod
    def get_skeleton_from_kinect(video_path):
        skeleton = read_skeleton(video_path, source='kinect')
        return skeleton

    def __get_hand_coordinates_from_skeleton(self, skeleton, source, modality):
        l_hand, r_hand = find_hand_coordinates(skeleton, use_hand_coords=False, dist2wrist=self.dist2wrist,
                                               source=source, modality=modality)
        return l_hand, r_hand

    def __get_hand_crops_from_frames(self, frames, skeleton):
        crop_frames = get_hand_crop_from_video(frames, skeleton, border=self.crop_border)
        return crop_frames

    def __get_hand_crops_from_flows(self, flows, skeleton):
        crop_flows = get_hand_crop_from_flow_seq(np.asarray(flows), skeleton, border=self.crop_border)
        return crop_flows

    def __get_hog_features(self, crop_frames):
        fds = extract_hog_from_video(crop_frames,
                                     num_orientations=self.num_orientations,
                                     cell_size=self.cell_size,
                                     block_size=self.block_size,
                                     visualize=self.visualize)
        return fds

    def __get_hof_features(self, crop_flows):
        fds = extract_hof_from_flow_seq(crop_flows,
                                        num_orientations=self.num_orientations + 1,
                                        cell_size=self.cell_size,
                                        block_size=self.block_size,
                                        visualize=self.visualize)
        return fds

    def __get_mbh_features(self, crop_flows):
        fds_x, fds_y = extract_mbh_from_flow_seq(crop_flows,
                                                 num_orientations=self.num_orientations,
                                                 cell_size=self.cell_size,
                                                 block_size=self.block_size,
                                                 visualize=self.visualize,
                                                 normalize=self.normalize)
        return fds_x, fds_y

    def save_descriptors(self, clazz, video, descriptors):
        filename = os.path.join(self.descriptors_path, clazz, video + '_desc.json')
        with open(filename, 'w') as fp:
            json.dump(descriptors, fp, sort_keys=True, indent=4)

    def load_descriptors(self, sample):
        with open(os.path.join(self.descriptors_path, sample[1], sample[0] + '_desc.json'), 'r') as fp:
            desc = json.load(fp)

        ds = DataSample()
        l_hog = np.asarray(desc['hog_left'])
        r_hog = np.asarray(desc['hog_right'])
        ds.hog = np.hstack((r_hog, l_hog))

        # l_t_hog = self.get_temporal_descriptors(l_hog)
        # r_t_hog = self.get_temporal_descriptors(r_hog)
        # ds.hog = np.hstack((r_t_hog, l_t_hog))

        l_hof = np.asarray(desc['hof_left'])
        r_hof = np.asarray(desc['hof_right'])
        ds.hof = np.hstack((r_hof, l_hof))

        # l_t_hof = self.get_temporal_descriptors(l_hof)
        # r_t_hof = self.get_temporal_descriptors(r_hof)
        # ds.hof = np.hstack((r_t_hof, l_t_hof))

        l_mbhx = np.asarray(desc['mbhx_left'])
        l_mbhy = np.asarray(desc['mbhy_left'])
        r_mbhx = np.asarray(desc['mbhx_right'])
        r_mbhy = np.asarray(desc['mbhy_right'])
        ds.mbh = np.hstack((r_mbhx, r_mbhy, l_mbhx, l_mbhy))

        # l_t_mbhx = self.get_temporal_descriptors(l_mbhx)
        # l_t_mbhy = self.get_temporal_descriptors(l_mbhy)
        # r_t_mbhx = self.get_temporal_descriptors(r_mbhx)
        # r_t_mbhy = self.get_temporal_descriptors(r_mbhy)
        # ds.mbh = np.hstack((r_t_mbhx, r_t_mbhy, l_t_mbhx, l_t_mbhy))

        return ds

    def prepare_classifier_data_for_descriptors(self):
        self.train_descriptors_hog = []
        self.train_descriptors_hof = []
        self.train_descriptors_mbh = []
        self.train_labels = []
        for ts in self.data_helper.training_split:
            ds = self.load_descriptors(ts)
            self.train_descriptors_hog.append(ds.hog)
            self.train_descriptors_hof.append(ds.hof)
            self.train_descriptors_mbh.append(ds.mbh)
            self.train_labels.append(ts[1])

        self.test_descriptors_hog = []
        self.test_descriptors_hof = []
        self.test_descriptors_mbh = []
        self.test_labels = []
        for ts in self.data_helper.test_split:
            ds = self.load_descriptors(ts)
            self.test_descriptors_hog.append(ds.hog)
            self.test_descriptors_hof.append(ds.hof)
            self.test_descriptors_mbh.append(ds.mbh)
            self.test_labels.append(ts[1])

    @staticmethod
    def get_fisher_vectors(data, pca, gmm):
        data_fv = []

        for d in data:
            pca_d = pca.transform(d)
            fv_d = generate_fisher_vector(pca_d, gmm['means'], gmm['covars'], gmm['priors'])
            data_fv.append(np.transpose(fv_d))

        return np.asarray(data_fv)

    def get_temporal_descriptors(self, descriptor):
        temporalStride = int(np.floor(self.length / self.ntCells))
        f_indices = np.linspace(0, descriptor.shape[0]-self.length-1, temporalStride * 1, dtype=np.int)

        features = []
        for f in f_indices:
            feature = descriptor[f:f+self.length]

            row = []
            pos = 0
            for i in range(0, self.ntCells):
                vec = np.asarray(sum(feature[pos:pos+temporalStride]))
                pos += temporalStride
                row.append(vec / temporalStride)

            row = np.reshape(np.asarray(row), self.ntCells * descriptor.shape[1])
            features.append(row)

        return np.asarray(features)