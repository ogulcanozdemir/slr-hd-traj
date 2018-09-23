import pickle
import numpy as np

data_root = '/raid/users/oozdemir/code/tm-shd-slr/experiments_hand/idt_hand_general_variance/idt_hand_hr45_l20_t3/'

hog_path = data_root + 'train_idt_hog_cat.pickle'
hof_path = data_root + 'train_idt_hof_cat.pickle'
mbh_path = data_root + 'train_idt_mbh_cat.pickle'

fisher_hog = data_root + 'k16/fisher_data_hog.pickle'
fisher_hof = data_root + 'k16/fisher_data_hof.pickle'
fisher_mbh = data_root + 'k16/fisher_data_mbh.pickle'


n_training_samples = 305

with open(fisher_hog, 'rb') as fhog:
    features_hog = pickle.load(fhog)

# n_features_hog, hog_dim =  features_hog.shape

with open(fisher_hof, 'rb') as fhof:
    features_hof = pickle.load(fhof)

# n_features_hof, hof_dim =  features_hof.shape

with open(fisher_mbh, 'rb') as fmbh:
    features_mbh = pickle.load(fmbh)

# n_features_mbh, mbh_dim =  features_mbh.shape

mx_features = np.max((n_features_hog, n_features_hof, n_features_mbh))
avg_features = mx_features / n_training_samples
sum_dim = hog_dim + hof_dim + mbh_dim

print('number of features per sign: {}, number of features: {}, feature size: {}'.format(avg_features, mx_features, sum_dim))