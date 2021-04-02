

import time
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader
from model import EncoderRNN, DecoderRNN

#!pip install einops
import einops

class StrokesDataset(Dataset):
    def __init__(self, dataset: np.array, max_seq_length: int, scale: Optional[float] = None):
        data = []
        for seq in dataset:
            if 10 < len(seq) <= max_seq_length:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)

                seq = np.array(seq, dtype=np.float32)
                data.append(seq)

        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale

        longest_seq_len = max([len(seq) for seq in data])

        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)

        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)

            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / scale
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]
            self.data[i, len_seq + 1:, 4] = 1
            self.mask[i, :len_seq + 1] = 1
        self.data[:, 0, 2] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.mask[idx]


class Sampler:
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        self.decoder = decoder
        self.encoder = encoder
    
    def sample(self, data: torch.Tensor, temperature: float):
        longest_seq_len = len(data)
        z, _, _ = self.encoder(data)

        s = data.new_tensor([0, 0, 1, 0, 0])
        seq = [s]

        state = None

        with torch.no_grad:
            for i in range(longest_seq_len):
                data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
                dist, q_logits, state = self.decoder(data, z, state)
                s = self._sample_step(dist, q_logits, temperature)
                seq.append(s)
                if s[4] == 1:
                    break
        seq = torch.stack(seq)

        self.plot(seq)
    
    @staticmethod
    def _sample_step(dist: 'BivarateGaussanMixture', q_logits: torch.Tensor, temperature:float):
        dist.set_temperature(temperature)
        pi, mix = dist.get_distribution()
        idx = pi.sample()[0, 0]
        q = torch.distributions.Categorical(logits=q_logits / temperature)
        q_idx = q.sample()[0, 0]
        xy = mix.sample()[0, 0, idx]
        stroke = q_logits.new_zeros(5)
        stroke[:2] = xy
        stroke[q_idx + 2] = 1

        return stroke

    @staticmethod
    def plot(seq: torch.Tensor):
        seq[:, 0:2] = torch.cumsum(seq[:, 0:2], dim=0)
        seq[:, 2] = seq[:, 3]
        seq = seq[:, 0:3].detach().cpu().numpy()
        strokes = np.split(seq, np.where(seq[:, 2]>0)[0] + 1)

        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        
        plt.axis('off')
        plt.show()
