import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Push through RNN layer (the ouput is irrelevant)
        _, self.hidden = self.lstm(inputs, self.hidden)
        return self.hidden


class Decoder(nn.Module):

    def __init__(self, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        # input_size=1 since the output are single values
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, outputs, hidden, criterion):
        batch_size, num_steps = outputs.shape
        # Create initial start value/token
        input = torch.tensor([[0.0]] * batch_size, dtype=torch.float)
        # Convert (batch_size, output_size) to (seq_len, batch_size, output_size)
        input = input.unsqueeze(0)

        loss = 0
        for i in range(num_steps):
            # Push current input through LSTM: (seq_len=1, batch_size, input_size=1)
            output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, 1)
            output = self.out(output[-1])
            # Generate input for next step by adding seq_len dimension (see above)
            input = output.unsqueeze(0)
            # Compute loss between predicted value and true value
            loss += criterion(output, outputs[:, i])
        return loss


if __name__ == '__main__':

    # 5 is the number of features of your data points
    encoder = Encoder(2, 3)
    decoder = Decoder(3)
    # Create optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Some toy data: 2 sequences of length 10 with 2 features for each data point
    inputs = [
        [
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
        ],
        [
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
            [0.5, 0.2],
        ]
    ]

    inputs = torch.tensor(np.array(inputs), dtype=torch.float)
    # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
    inputs = inputs.transpose(1,0)

    # 2 sequences (to match the batch size) of length 6 (for the 6h into the future)
    outputs = [ [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.3, 0.2, 0.1] ]
    outputs = torch.tensor(np.array(outputs), dtype=torch.float)

    #
    # Do one complete forward & backward pass
    #
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Reset hidden state of encoder for current batch
    encoder.hidden = encoder.init_hidden(inputs.shape[1])
    # Do forward pass through encoder
    hidden = encoder(inputs)
    # Do forward pass through decoder (decoder gets hidden state from encoder)
    loss = decoder(outputs, hidden, criterion)
    # Backpropagation
    loss.backward()
    # Update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()
    print("Loss:", loss.item())












import cv2, torch, os, datetime
import numpy as np
from torch import nn

import multiprocessing
from os import getpid
import argparse

from dataset import CustomDataset
from models import ModifiedUNet


def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('-debug', default=True, type=arg_to_bool)
parser.add_argument('-device', default='cpu')
parser.add_argument('-z_size', default=16, help='Size of embedding for decoder', type=int)

parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-accumulator_size', default=10, type=int, help='How many items can be held at max in sequence container')
parser.add_argument('-sequence_window', default=3, type=int, help='Sequence length, this is a used as a sliding window')
parser.add_argument('-epoch', default=1000, type=int)

parser.add_argument('-dataset_size', default=100, type=int, help='How many sequences are generated, higher number increases RAM')
parser.add_argument('-num_workers', default=0, type=int, help='How many parralel workers on dataloader')
parser.add_argument('-beta', default=1.0, type=float, help='Beta hyperparameter in beta VAE')
parser.add_argument('-image_size', choices=[64], default=64, type=int, help='Generated image size')
parser.add_argument('-lstm_layers', type=int, help='How many layers in LSTM', default=1)

parser.add_argument('-vx', type=float, default=4, help='Velocity along x-axis')
parser.add_argument('-vy', type=float, default=0, help='Velocity along y-axis')
parser.add_argument('-g', type=float, default=1, help='Acceleration due to gravity')
parser.add_argument('-r', type=float, default=5, help='Radius of generated image circle')

parser.add_argument('-load', default='', help='Specify save folder name and the last epoch will be tanken')
args = parser.parse_args()

def show(truth_t, pred_t):
    img_t = torch.cat((truth_t, pred_t), dim=2)
    img_t = img_t.permute(1, 2, 0)
   
    img = img_t.cpu().detach().numpy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(1)
   

def load(path):
    if not os.path.exists(path):
        print('Load path doesnt exist!')
    else:
        files = [f for f in os.listdir(path) if os.isfile(os.path.join(path, f))].sort()
    
    return files

# dataloader returns differently sized sequences
# collate_fn overloads th default behaviour of pytorch batch stacking
def collate_fn(data):

    def pack(data, key):
        idxs = []
        idx_from = 0
        imgs_batch = []

        for sample in data:
            imgs = sample[key]
            imgs_batch.append(imgs)

            idx_to = idx_from + imgs.shape[0]
            idxs.append({"idx_from": idx_from, "idx_to": idx_to})
            idx_from = idx_to
        return imgs_batch, idxs
    
    imgs, idxs = pack(data, 'imgs')
    imgs_truths, idxs_truths = pack(data, 'imgs_truths')

    # 2D sequences to 1D array
    batch_1D = torch.cat(imgs)
    batch_truths_1D = torch.cat(imgs_truths)

    return batch_1D, idxs, batch_truths_1D, idxs_truths


print('Using device:', args.device)

model = ModifiedUNet(args, in_channels=1, out_channels=1, bottleneck_out=None, init_features=32).to(args.device)


now = datetime.datetime.now()
dir_name = now.strftime("%B_%d_at_%H_%M_%p")
save_dir = './save/' + dir_name

optimizer = torch.optim.Adam(model.parameters(), args.lr)
reconstruction_loss_fn = torch.nn.BCELoss()

dataset = CustomDataset(args)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

for epoch in range(1, args.epoch):
    epoch_loss_rec = []
    epoch_loss_kl = []

    for batch, idxs, batch_truths, idxs_truths in dataset_loader:
        batch = batch.to(args.device)
        
        preds = model.forward(batch, idxs)

        

        decoded = decoder.forward(lstm_last_stacked)

        if args.debug:
            show(last_frames_t[0], decoded[0])
        
        if args.load:
            continue

        loss_recunstruction = reconstruction_loss_fn(decoded, last_frames_t)

        loss_kl = args.beta * 0.5 * (1.0 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)
        loss_kl = torch.mean(loss_kl)
        loss = loss_recunstruction - loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_kl.append(float(loss_kl))
        epoch_loss_rec.append(float(loss_recunstruction))



    epoch_loss_kl = np.average(np.array(epoch_loss_kl))
    epoch_loss_rec = np.average(np.array(epoch_loss_rec))
    print(f'epoch: {epoch}   |    kl: {epoch_loss_kl:.4}    |    rec: {epoch_loss_rec:.4}')        

    if epoch % 50 == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': encoder.state_dict(),
                'lstm': lstm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            }, save_dir + f'/{epoch}_{float(loss)}_save.pth')
