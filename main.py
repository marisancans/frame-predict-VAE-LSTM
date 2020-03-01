import cv2, torch, os, datetime
import numpy as np
from torch import nn

import multiprocessing
from os import getpid
import argparse

from dataset import CustomDataset
from models import Encoder, Decoder, LSTMModel


def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('-debug', default=True, type=arg_to_bool)
parser.add_argument('-device', default='cpu')
parser.add_argument('-z_size', default=16, help='Size of embedding for decoder', type=int)

parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epoch', default=1000, type=int)

parser.add_argument('-dataset_size', default=100, type=int, help='How many sequences are generated, higher number increases RAM')
parser.add_argument('-num_workers', default=0, type=int, help='How many parralel workers on dataloader')
parser.add_argument('-beta', default=1.0, type=float, help='Beta hyperparameter in beta VAE')
parser.add_argument('-image_size', choices=[3, 7, 15, 31, 63, 127, 255, 511], default=63, type=int, help='Generated image size')
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
def collate_fn(batch):
    idxs = []
    last_frames = []
    idx_from = 0
    imgs_batch = []

    for s in batch:
        imgs = s['imgs']
        last_frames.append(s['last_frame'])
        imgs_batch.append(imgs)

        idx_to = idx_from + imgs.shape[0]
        idxs.append({"idx_from": idx_from, "idx_to": idx_to})
        idx_from = idx_to
    
    # 2D sequences to 1D array
    batch_1D = torch.cat(imgs_batch)
    last_frames_t = torch.stack(last_frames)

    return batch_1D, last_frames_t, idxs


print('Using device:', args.device)

encoder = Encoder(args).to(args.device)
decoder = Decoder(args).to(args.device)
lstm = LSTMModel(input_size=args.z_size, hidden_size=args.z_size, num_layers=args.lstm_layers, args=args).to(args.device)

now = datetime.datetime.now()
dir_name = now.strftime("%B_%d_at_%H_%M_%p")
save_dir = './save/' + dir_name

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, args.lr)
reconstruction_loss_fn = torch.nn.BCELoss()

dataset = CustomDataset(args)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

for epoch in range(1, args.epoch):
    epoch_loss_rec = []
    epoch_loss_kl = []

    for batch_t, last_frames_t, idxs in dataset_loader:
        batch_t = batch_t.to(args.device)
        last_frames_t = last_frames_t.to(args.device)
        
        z_vector, z_mu, z_sigma = encoder.forward(batch_t)

        # 1D to 2D sequences
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
