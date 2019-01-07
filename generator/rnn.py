import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




images = []
# for root, dirs, files in os.walk("'/home/maris/rnn/'"):
#     [images.append(f) for f in files if os.path.isfile(f) not in False]
path = '/home/maris/rnn/'
for dir in os.listdir(path):
    if os.path.isdir(path + dir):
        d = []
        for f in os.listdir(path + dir):
            if os.path.isfile(path + dir + '/' + f):
                d.append(path + dir + '/' + f)
        images.append(d)



data = Dataset(images)