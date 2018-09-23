import os
from pandas import *
import matplotlib.pyplot as plt
import subprocess
import re
from decimal import Decimal


def get_video_length(path):
    process = subprocess.Popen(['/raid/users/oozdemir/tools/ffmpeg/ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', '-i', path],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()

    return float(stdout)

datapath = '/raid/users/oozdemir/data/BosphorusSign/ToyDataset'
classdir = sorted(list(map(int, os.listdir(datapath))))

numbers = {}
for _clazz in classdir:
    numbers[_clazz] = 0

dur = []
for _clazz in classdir:
    _clazzpath = os.path.join(datapath, str(_clazz))
    userdir = sorted(os.listdir(_clazzpath))
    for _us in userdir:
        dur.append(get_video_length(os.path.join(datapath, str(_clazz), str(_us), 'color_scaled.mp4')))

mean = np.mean(dur, axis=0)
print(mean)

# for k, v in numbers.items():
#     print(v, end='\n')
#
#




# fig = plt.figure(figsize=(8,12))
# rects = plt.barh(range(numbers.__len__()),
#                  numbers.items(),
#                  height=0.8,
#                  align="center",
#                  color="#8A0707",
#                  edgecolor="none")