import cv2
import torch
from torch.utils.data.dataset import Dataset
import random as random
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        height, width = args.image_size, args.image_size
        self.data = []
        self.index = []

        print(f'Generating samples...')

        for i in range(args.dataset_size):
            
            x, y, t = 0, 0, 0
            c = 0
            offset = random.randint(1, int(height/2))
            seq_idx = {
                "idx_from": len(self.data),
                "idx_to": 0
            }

            while True:
                t += 1
                img = np.zeros((height, width, 1), np.float32)
                img.fill(1)
                img = cv2.circle(img, (int(x), int(abs(y)) + args.r + offset), args.r, (0), -1)
                               
                self.data.append(img)
                x = args.vx * t
                y = args.vy * t - 1 / 2 * args.g * t ** 2

                if abs(y) > height - args.r - offset:
                    break

                c += 1

            seq_idx["idx_to"] = len(self.data)
            self.index.append(seq_idx)

        
    def __getitem__(self, idx):
        i = self.index[idx]
        imgs = self.data[i["idx_from"] : i["idx_to"] - 1]
        last_frame = self.data[i["idx_to"] - 1]
        
        imgs = torch.FloatTensor(imgs)
        imgs = imgs.permute(0, 3, 1, 2) # (B, H, W, C) -->  (B, C, H, W)

        return imgs

    def __len__(self):
        return len(self.index)