from cyvlfeat.fisher import fisher

import numpy as np


def generate_fisher_vector(x, _means, _covariances, _priors, _improved=True, _normalized=False):
    return fisher(np.transpose(x.astype(dtype='float32')),
                  means=np.transpose(_means.astype(dtype='float32')),
                  covariances=np.transpose(_covariances.astype(dtype='float32')),
                  priors=np.transpose(_priors.astype(dtype='float32')),
                  improved=_improved,
                  normalized=_normalized)


def generate_vlad(x):
    pass


def generate_bof(x):
    pass
