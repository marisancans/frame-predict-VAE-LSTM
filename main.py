import cv2, torch, os, datetime
import numpy as np
from torch import nn
import torchvision
import argparse
import os
from dataset import PymunkStreamDataset
from models import ModifiedUNet
from radam import RAdam

def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('-debug', default=True, type=arg_to_bool)
parser.add_argument('-device', default='cpu')
parser.add_argument('-z_size', default=16, help='Size of embedding for decoder', type=int)

parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-sequence_length', default=10, type=int)
parser.add_argument('-sequence_window', default=3, type=int, help='Sequence length, this is a used as a sliding window')
parser.add_argument('-epoch', default=100000, type=int)

parser.add_argument('-dataset_size', default=256, type=int, help='How many sequences are generated, higher number increases RAM')
parser.add_argument('-shuffle', default=False, type=arg_to_bool, help='Wether to shuffle the dataset after each epoch, not shuffilg so i can create pretty gifs')
parser.add_argument('-num_workers', default=0, type=int, help='How many parralel workers on dataloader')
parser.add_argument('-beta', default=1.0, type=float, help='Beta hyperparameter in beta VAE')
parser.add_argument('-image_size', choices=[64], default=64, type=int, help='Generated image size')
parser.add_argument('-lstm_layers', type=int, help='How many layers in LSTM', default=1)

parser.add_argument('-loss', choices=['mse', 'bce'], default='mse')
parser.add_argument('-optimizer', choices=['adam', 'radam'], default='adam')
parser.add_argument('-autoencoder_type', choices=['variational', 'vanilla'], default='vanilla')
parser.add_argument('-load_path', help='Specify save folder name and the last epoch will be tanken', type=str)
parser.add_argument('-grid_latent', help='If load_path arg specified, load model and create grid from z latent', default=False, type=arg_to_bool)

args = parser.parse_args()

def walk_grid(model):
    model.eval()



def show(truth_t, pred_t, imgs_dir, epoch):
    imgs = []
    for t, p in zip(truth_t, pred_t):
        imgs.append(t)
        imgs.append(p)
    imgs = torch.stack(imgs)

    grid_t = torchvision.utils.make_grid(imgs, nrow=16)
   
    img = grid_t.cpu().permute(1, 2, 0).detach().numpy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(1)
    cv2.imwrite(imgs_dir + '/' + str(epoch) + '.png', img * 255)



print('Using device:', args.device)

model = ModifiedUNet(args, in_channels=1, out_channels=1, bottleneck_out=None, init_features=32).to(args.device)

now = datetime.datetime.now()
dir_name = now.strftime("%B_%d_at_%H_%M_%p")
save_dir = './save/' + dir_name
imgs_dir = './imgs/' + dir_name

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
else:
    optimizer = RAdam(model.parameters(), args.lr)

if args.loss == 'mse':
    loss_fn = torch.nn.MSELoss()
else:
    loss_fn = torch.nn.BCELoss()

# laoding checkpoint
if args.load_path:
    files = os.listdir(args.load_path)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    last_path = os.path.join(args.load_path, files[-1])

    checkpoint = torch.load(last_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']

    
if args.grid_latent:
    walk_grid(model)
    os._exit(0)


dataset = PymunkStreamDataset(args.batch_size, args.sequence_window)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=args.batch_size)

for epoch in range(1, args.epoch):
    for frames_t, truths_t in dataset_loader:
        frames_t = frames_t.to(args.device)
        truths_t = truths_t.to(args.device)
        
        # remove batch dim created by torch dataset
        frames_t = frames_t.squeeze(0).float()
        truths_t = truths_t.squeeze(0).float()

        # swap axis
        decoded, z_mu, z_sigma = model.forward(frames_t)
        
        if args.debug:
            if not os.path.exists(imgs_dir):
                os.makedirs(imgs_dir)
            show(truths_t.detach().cpu(), decoded.detach().cpu(), imgs_dir, epoch)

        loss_recunstruction = loss_fn(decoded, truths_t)
        
        if args.autoencoder_type == 'variational':
            loss_kl = args.beta * 0.5 * (1.0 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)
            loss_kl = torch.mean(loss_kl)
            loss = loss_recunstruction - loss_kl
            epoch_loss_kl.append(float(loss_kl))
            epoch_loss_rec.append(float(loss_recunstruction))
        else:
            loss = loss_recunstruction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f'epoch: {epoch}   |    loss: {loss:.4}')        

    if epoch % 50 == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            }, save_dir + f'/{epoch}.pth')
