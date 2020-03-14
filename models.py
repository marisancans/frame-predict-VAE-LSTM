import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

def conv_output_shape(h_w, kernel_size=(1, 1), stride=(1, 1), pad=(0, 0), dilation=1):   
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    return h, w

def convtransp_output_shape(h_w, kernel_size=(1, 1), stride=(1, 1), pad=(0, 0), dilation=1):       
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]
    return h, w

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', num_layers=1):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.device = device

    def forward(self, x):
        # Initialize hidden state with zeros
        S, B, z = x.shape

        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)
        

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)
        all_timesteps, hidden_last_timestep = self.lstm(x, (h0, c0))

        return hidden_last_timestep

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', num_layers=1):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.device = device

    def forward(self, sizes, steps, hidden):
        S, B, z = sizes
        input = torch.zeros([max(steps), B, z], dtype=torch.float).to(self.device)
        preds, hidden = self.lstm(input, hidden)

        preds = preds.permute(1, 0, 2) # (Seq_len, B, z) --> (B, Seq_len, z)
        return preds

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
        self.bn_enc1 = nn.BatchNorm2d(features)
        self.encoder2 = ModifiedUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc2 = nn.BatchNorm2d(features * 2)
        self.encoder3 = ModifiedUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc3 = nn.BatchNorm2d(features * 4)
        self.encoder4 = ModifiedUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_enc4 = nn.BatchNorm2d(features * 8)

        # bottleneck
        self.bottleneck = ModifiedUNet._block(features * 8, self.bottleneck_out, name="bottleneck")
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
       
        # VAE
        self.mu = torch.nn.Linear(in_features=self.bottleneck_out, out_features=self.args.z_size)
        self.sigma = torch.nn.Linear(in_features=self.bottleneck_out, out_features=self.args.z_size)

        # LSTM
        self.lstm_encoder = LSTMEncoder(input_size=args.z_size, hidden_size=args.z_size, device=args.device).to(args.device)
        self.lstm_decoder = LSTMDecoder(input_size=args.z_size, hidden_size=args.z_size, device=args.device).to(args.device)  

        # decoder
        
        # calculate what shape should be before upconv4
        self.upconv6 = nn.ConvTranspose2d(
            args.z_size, features * 16, kernel_size=2, stride=2
        )
        self.decoder6 = ModifiedUNet._block(features * 16, features * 16, name="dec6")
        self.bn_dec6 = nn.BatchNorm2d(features * 16)

        self.upconv5 = nn.ConvTranspose2d(
            features * 16, features * 16, kernel_size=2, stride=2
        )
        self.decoder5 = ModifiedUNet._block(features * 16, features * 16, name="dec5")
        self.bn_dec5 = nn.BatchNorm2d(features * 16)


        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ModifiedUNet._block(features * 8, features * 8, name="dec4")
        self.bn_dec4 = nn.BatchNorm2d(features * 8)


        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ModifiedUNet._block(features * 4, features * 4, name="dec3")
        self.bn_dec3 = nn.BatchNorm2d(features * 4)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ModifiedUNet._block(features * 2, features * 2, name="dec2")
        self.bn_dec2 = nn.BatchNorm2d(features * 2)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ModifiedUNet._block(features, features, name="dec1")
        self.bn_dec1 = nn.BatchNorm2d(features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )



    def forward(self, x, idxs, steps):
        enc1 = self.encoder1(x)
        enc1 = self.bn_enc1(enc1)
        enc1 = F.relu(enc1)

        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.bn_enc2(enc2)
        enc2 = F.relu(enc2)

        enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc3)
        enc3 = self.bn_enc3(enc3)
        enc3 = F.relu(enc3)

        enc4 = self.pool3(enc3)
        enc4 = self.encoder4(enc4)
        enc4 = self.bn_enc4(enc4)
        enc4 = F.relu(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))
        avg_pooled = self.avg_pool(bottleneck)
        avg_pooled = avg_pooled.squeeze(-1).squeeze(-1)
                
        # VAE
        z_mu = self.mu.forward(avg_pooled)
        z_sigma = self.sigma.forward(avg_pooled)

        bs = x.shape[0]
        eps = torch.randn(bs, self.args.z_size).to(self.args.device) * z_sigma + z_mu # Sampling epsilon from normal distributions
        
        
        z_vector = z_mu + z_sigma * eps # z ~ Q(z|X)

        # 1D to 2D
        z_sequences = [z_vector[i['idx_from'] : i['idx_to']] for i in idxs]
        z_sequences = torch.stack(z_sequences)
        z_sequences = z_sequences.permute(1, 0, 2) # (B, Seq_len, z_vector) --> (Seq_len, B, z_vector)
 
        # encode time, return last lastm hidden state
        hidden_last_timestep = self.lstm_encoder.forward(z_sequences)

        # decode and predict time 
        preds = self.lstm_decoder.forward(z_sequences.shape, steps, hidden_last_timestep)
        
        # pick predictions that we need
        picked_preds = []
        for batch_idx, to_step in enumerate(steps):
            picked_preds.append(preds[batch_idx][0:to_step])

        masked_catted = torch.cat(picked_preds, dim=0)           

        # add H and W dims
        masked_catted = masked_catted.unsqueeze(-1).unsqueeze(-1)

        expand = self.upconv6(masked_catted)
        expand = self.decoder6(expand)
        expand = self.bn_dec6(expand)
        expand = F.relu(expand)
        expand = self.upconv5(expand)
        expand = self.decoder5(expand)
        expand = self.bn_dec5(expand)
        expand = F.relu(expand)

        dec4 = self.upconv4(expand)
        dec4 = self.decoder4(dec4)
        dec4 = self.bn_dec4(dec4)
        dec4 = F.relu(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)
        dec3 = self.bn_dec3(dec3)
        dec3 = F.relu(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)
        dec2 = self.bn_dec2(dec2)
        dec2 = F.relu(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        dec1 = self.bn_dec1(dec1)
        dec1 = F.relu(dec1)

        out = self.conv(dec1)
        out = torch.sigmoid(out)
        return out, z_mu, z_sigma


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


