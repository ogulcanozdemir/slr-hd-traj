from sklearn.decomposition import PCA


def pca(x, var_explained=0.99, zero_mean=True):
    return PCA(n_components=var_explained,
               svd_solver='full',
               whiten=zero_mean).fit(x)
