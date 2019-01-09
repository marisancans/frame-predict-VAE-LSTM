import torch.nn.functional as F
import torch.nn as nn

import numpy as np

# FOR LSTM function
# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers – Number of recurrent layers. E.g., setting num_layers=2
# would mean stacking two LSTMs together to form a stacked LSTM,
# with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1


# FOR CNN function
# out_channel represents how many feature maps we want, think of this as a heatmap

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 8 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, 3)
        # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # torch.nn.ReLU()
        self.lstm1 = nn.LSTM(8, 16, 2)
        self.linear1 = nn.Linear(16, 8)
        self.convtranspose1 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3)  # Deconvolution
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lstm1(x)
        x = self.linear1(x)
        x = self.convtranspose1(x)
        x = self.output(x)
        return x
