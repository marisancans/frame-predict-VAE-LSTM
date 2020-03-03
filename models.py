import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, args):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.args = args

    def forward(self, x, bs):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.args.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.args.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        return out

# https://github.com/mateuszbuda/brain-segmentation-pytorch
class ModifiedUNet(nn.Module):

    def __init__(self, args, in_channels=1, out_channels=1, bottleneck_out=None, init_features=32):
        super(ModifiedUNet, self).__init__()

        features = init_features

        features = init_features
        self.bottleneck_out = bottleneck_out
        self.args = args

        if not bottleneck_out:
            self.bottleneck_out = features * 16

        # encoder
        self.encoder1 = ModifiedUNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ModifiedUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ModifiedUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ModifiedUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ModifiedUNet._block(features * 8, self.bottleneck_out, name="bottleneck")
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
       
        # VAE
        self.mu = torch.nn.Linear(in_features=self.bottleneck_out, out_features=self.args.z_size)
        self.sigma = torch.nn.Linear(in_features=self.bottleneck_out, out_features=self.args.z_size)

        # LSTM
        lstm = LSTMModel(input_size=args.z_size, hidden_size=args.z_size, num_layers=args.lstm_layers, args=args).to(args.device)

        # decoder
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ModifiedUNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ModifiedUNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ModifiedUNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ModifiedUNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, idxs):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        z_vector = self.avg_pool(bottleneck)
        z_vector = z_vector.squeeze(-1).squeeze(-1)
                
        # VAE
        z_mu = self.mu.forward(z_vector)
        z_sigma = self.sigma.forward(z_vector)

        bs = x.shape[0]
        eps = torch.randn(bs, self.args.z_size).to(self.args.device) * z_sigma + z_mu # Sampling epsilon from normal distributions
        
        
        z_vector = z_mu + z_sigma * eps # z ~ Q(z|X)

        # LSTM + packing

        # accumulator_size
        # sequence_window
        z_sequences = [z_vector[i['idx_from'] : i['idx_to']] for i in idxs]

        # run through LSTM, use packing
        padded_seq = nn.utils.rnn.pad_sequence(z_sequences, batch_first=True)
        lengths = [x.size(0) for x in z_sequences]
        pack = nn.utils.rnn.pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)

        lstm_out = lstm.forward(pack, bs=len(z_sequences))
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # take last LSTM output
        lstm_last = [seq[idx] for seq, idx in zip(unpacked, unpacked_len-1)]
        lstm_last_stacked = torch.stack(lstm_last)

        

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


