import numpy as np
import pickle


file = 'fhd_crop80_ncell40_nblock2_nbins8.pickle'
with open(file, 'rb') as fp:
    feature = pickle.load(fp)


nt = 3

rpt = np.mod(f.shape[0], nt)
dup_rows_avg = np.mean(f[-rpt:], axis=0)
f = np.vstack((f[0:-rpt], np.repeat([dup_rows_avg], repeats=nt, axis=0)))
indices = np.arange(0, f.shape[0]).reshape([int(f.shape[0]/nt), nt])
new_f = []
[new_f.append(np.mean(f[idx, :], axis=0)) for idx in indices]

print(feature)
