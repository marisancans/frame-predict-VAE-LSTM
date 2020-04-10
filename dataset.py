import cv2
import torch
from torch.utils.data.dataset import Dataset
import random as random
import numpy as np
import matplotlib.pyplot as plt
from bouncy_balls import BouncyBalls

class PymunkStreamDataset(torch.utils.data.Dataset):
    def __init__(self, length, sequence_window):
        super(PymunkStreamDataset).__init__()
        self.length = length
        self.sequence_window = sequence_window

        self.game = BouncyBalls()

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        frames = self.game.get_frames(self.length)
        frames = [x.astype(np.uint8) for x in frames]
        frames = [cv2.flip(x, flipCode=1) for x in frames]
        frames = [np.rot90(x) for x in frames]
        frames = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frames]
        frames = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in frames]
        frames = [cv2.resize(x, (64, 64)) for x in frames]
        frames = [x / 255.0 for x in frames]
        frames = [torch.tensor(x) for x in frames]
        frames = [x.unsqueeze(0) for x in frames]
        frames_t = torch.stack(frames).float()

        seq = []
        truths = []
        
        # sequence_window
        for i in range(0, len(frames) - self.sequence_window):
            t = frames[i + self.sequence_window]
            truths.append(t)

        truths_t = torch.stack(truths)


        return frames_t, truths_t