import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import os, io, cv2

from net import Net
from dataset import Dataset

#   Settings
img_path = '/home/maris/rnn/'


#   Load all image paths
data = []

for directory in os.listdir(img_path):
    if os.path.isdir(img_path + directory):
        x = []

        # Sort via filenames, because python appends unordered and it will cause LSTM network confusion
        files = sorted(os.listdir(img_path + directory))

        is_file = lambda name : os.path.isfile(name)
        prefix = lambda path : img_path + directory + '/' + path

        # X is sequence of frames, except for the last one
        for f in files:
            if is_file(prefix(f)):
                x.append(prefix(f))

        data.append(x)


data = Dataset(data)


dataset_train = Dataset(data=data[0:int(len(data) * 0.8)])
dataset_test = Dataset(data=data[int(len(data) * 0.8):])

batch_size = 16
max_epochs = 10

data_loder_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True
)

data_loder_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False
)


model = Net()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


losses_train = []


for epoch in range(max_epochs):
    for data_loader in data_loder_train:
        losses_tmp = []

        for idx, batch in enumerate(data_loader[:-1]):
            x_paths = batch
            y_paths = data_loader[idx + 1]
            X = [cv2.imread(i, 0) for i in x_paths] #  Mode 1 is grayscale
            Y = [cv2.imread(i, 0) for i in y_paths]
            X = torch.tensor(X)

            #TODO implement batching here, X is (batchsize, width, height)

            y_prim = model.forward(X)