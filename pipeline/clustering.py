from cyvlfeat.gmm import gmm
from cyvlfeat.kmeans import kmeans


def generate_gmms(X, _clusters=64):
    means, covars, priors, ll, posteriors = gmm(X,
                                                n_clusters=_clusters,
                                                max_num_iterations=10000,
                                                verbose=True)

    model = {
        'priors': priors,
        'means' : means,
        'covars': covars,
        'll'    : ll,
        'posteriors': posteriors
    }

    return model


def generate_kmeans(X, _clusters=32):
    centers = kmeans(X,
                     num_centers=_clusters,
                     max_num_iterations=10000,
                     verbose=True)
    return centers