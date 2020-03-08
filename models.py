import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
class ManyToManyLSTMM(nn.Module):
    def __init__(self, input_size, hidden_size, args):
        super(ManyToManyLSTMM, self).__init__()

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.args = args

    def forward(self, x):
        S, B, z = x.shape

        # Initialize cell state
        hx = torch.randn(B, self.hidden_size)
        cx = torch.randn(B, self.hidden_size)

        # Run through first n known frames
        for i in range(S):
            hx, cx = self.cell(x[i], (hx, cx))

        # Try to predict n frames
        output = []
        for i in range(2):
            hx, cx = self.cell(hx, (hx, cx))
            output.append(hx)

        return output

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
        self.lstm = ManyToManyLSTMM(input_size=args.z_size, hidden_size=args.z_size, args=args).to(args.device)

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

    def lstm_predict(self, z_sequences):
        lstm_out = self.lstm.forward(z_sequences)

        # take 
        lstm_last = [seq[idx] for seq, idx in zip(unpacked, unpacked_len-1)]
        lstm_last_stacked = torch.stack(lstm_last)
        x=0

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

        # 1D to 2D
        z_sequences = [z_vector[i['idx_from'] : i['idx_to']] for i in idxs]
        z_sequences = torch.stack(z_sequences)
        z_sequences = z_sequences.permute(1, 0, 2) # (B, Seq_len, z_vector) --> (Seq_len, B, z_vector)
 
        # get next N frames from LSTM
        next_pred = self.lstm_predict(z_sequences)

        

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


