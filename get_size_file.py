import os
import numpy as np

data_path = '/raid/users/oozdemir/test/ToyDataset'

sizes = []

classdir = os.listdir(data_path)
for c in classdir:

    video_dir = os.listdir(os.path.join(data_path, c))
    for v in video_dir:
        if os.path.isfile(os.path.join(data_path, c, v, 'color.features')):
            file_siz = os.path.getsize(os.path.join(data_path, c, v, 'color.features'))
            file_siz = file_siz / (1024*1024.0)
            print('Size of {}:{} is {}'.format(c, v, file_siz))
            sizes.append(file_siz)

print('mean size is {}'.format(np.mean(sizes, axis=0)))