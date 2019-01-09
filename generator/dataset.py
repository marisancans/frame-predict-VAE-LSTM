import torch


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

    def showSample(self):
        fig=plt.figure(figsize=(8, 8))
        for i in range(0, len(self.data[0])):
            l = sorted(self.data[0])
            img = plt.imread(l[i])
            fig.add_subplot(1, len(self.data[0]), i+1)
            plt.imshow(img)
        plt.show()

