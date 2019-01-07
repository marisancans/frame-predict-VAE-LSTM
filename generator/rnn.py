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
        images.append(sorted(d))

# fig=plt.figure(figsize=(8, 8))
# for i in range(0, len(images[0])):
#     l = sorted(images[0])
#     img = plt.imread(l[i])
#     fig.add_subplot(1, len(images[0]), i+1)
#     plt.imshow(img)
# plt.show()

data = Dataset(images)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input):
        return self.layers.forward(input)


model = Model()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


