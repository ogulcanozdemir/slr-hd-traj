from data.split import load_split
from pipeline.features.idt_extractor import IdtExtractor
from pipeline.dimensionality_reduction import pca
from pipeline.clustering import generate_gmms

from data.data_helper import ToyDataHelper
from parameter_parser import ParameterParser

from os.path import sep

import os


if __name__ == '__main__':

    params = ParameterParser('idt').params
    data_helper = ToyDataHelper(params)
    extractor = IdtExtractor(params, data_helper)
    extractor.prepare_features()
    extractor.set_features(return_dict=True)

    print(1)