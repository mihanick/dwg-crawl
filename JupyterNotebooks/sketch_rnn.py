'''
    taken from https://nn.labml.ai/sketch_rnn/index.html
'''

import time
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader

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

class BivariateGaussianMixture:
    def __init__(self, pi_logits: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor):
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    @property
    def n_distributions(self):
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        self.pi_logits /= temperature
        self.sigma_x *= math.sqrt(temperature)
        self.sigma_y *= math.sqrt(temperature)

    def get_distribution(self):
        sigma_x = torch.clamp_min(self.sigma_x, 1e-5)
        sigma_y = torch.clamp_min(self.sigma_y, 1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1-1e-5)

        mean = torch.stack([self.mu_x, self.mu_y], -1)

        cov = torch.stack([
            sigma_x * sigma_x, rho_xy*sigma_x*sigma_y,
            rho_xy*sigma_x*sigma_y, sigma_y*sigma_y
        ], -1)
        cov = cov.view(*sigma_y.shape, 2, 2)

        multi_dist = torch.distributions.MultivariateNormal(
            mean, covariance_matrix=cov)

        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)

        return cat_dist, multi_dist

class EncoderRNN(torch.nn.Module):
    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        self.mu_head = nn.Linear(2*enc_hidden_size, d_z)
        self.sigma_head = nn.Linear(2*enc_hidden_size, d_z)

    def forward(self, inputs: torch.Tensor, state=None):
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')
        mu = self.mu_head(hidden)
        sigma_hat = self.sigma_head(hidden)
        sigma = torch.exp(sigma_hat / 2.)
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))

        return z, mu, sigma_hat

class DecoderRNN(torch.nn.Module):
    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int):
        super().__init__()

        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)
        self.init_state = nn.Linear(d_z, 2*dec_hidden_size)
        self.mixtures = nn.Linear(dec_hidden_size, 6*n_distributions)
        self.q_head = nn.Linear(dec_hidden_size, 3)
        self.q_log_softmax = nn.LogSoftmax(-1)
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size

    def forward(self, x: torch.Tensor, z: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        if state is None:
            h, c = torch.split(torch.tanh(self.init_state(z)),
                               self.dec_hidden_size, 1)
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())

        outputs, state = self.lstm(x, state)

        q_logits = self.q_log_softmax(self.q_head(outputs))

        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(
            self.mixtures(outputs), self.n_distributions, 2)

        dist = BivariateGaussianMixture(pi_logits, mu_x, mu_y, torch.exp(sigma_x), torch.exp(sigma_y), torch.tanh(rho_xy))

        return dist, q_logits, state

class ReconstructionLoss(torch.nn.Module):
    def forward(self, mask: torch.Tensor, target: torch.Tensor, dist: 'BivarateGaussianMixture', q_logits: torch.Tensor):
        pi, mix = dist.get_distribution()

        # target has shaep [sel_en, batch_size, 5(i.e. dx, dy, p1, p2, p3)] 
        # we want to get dx, dy and get the probabilities from each of distributions in the mixture
        # xy will have shape [seq_length, batch_size, n_distributions, 2]
        xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)

        probs = torch.sum(pi.probs * torch.exp(mix.log_prob(xy)), 2)

        loss_stroke = -torch.mean(mask * torch.log(1e-5 + probs))

        loss_pen = - torch.mean(target[:, :, 2:] * q_logits)

        return loss_stroke + loss_pen

class KLDivLoss(torch.nn.Module):
    def forward(self, sigma_hat: torch.Tensor, mu: torch.Tensor):
        return -0.5 * torch.mean(1 + sigma_hat - mu**2 - torch.exp(sigma_hat))

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

def main():
    enc_hidden_size = 256
    dec_hidden_size = 512

    batch_size = 100

    d_z = 128
    n_distributions = 20
    kl_div_loss_weight = 0.5
    grad_clip = 1.0
    temperature = 0.4
    max_seq_length = 200
    epochs = 100

    device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')

    encoder = EncoderRNN(d_z, enc_hidden_size).to(device)
    decoder = DecoderRNN(d_z, dec_hidden_size, n_distributions).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    sampler = Sampler(encoder, decoder)

    path = 'cat.npz'
    dataset = np.load(str(path), encoding='latin1', allow_pickle=True)
    
    train_dataset = StrokesDataset(dataset['train'], max_seq_length)
    valid_dataset = StrokesDataset(dataset['valid'], max_seq_length, train_dataset.scale)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size)

    start = time.time()

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        for i, batch in enumerate(train_loader):
            data = batch[0].to(device).transpose(0, 1)
            mask = batch[1].to(device).transpose(0, 1)

            z, mu, sigma_hat = encoder(data)

            z_stack = z.unsqueeze(0).expand(data.shape[0]- 1, -1, -1)

            inputs = torch.cat([data[:-1], z_stack], 2)

            dist, q_logits, _ = decoder(inputs, z, None)

            kl_loss = KLDivLoss()(sigma_hat, mu)
            reconstruction_loss = ReconstructionLoss()(mask, data[1:], dist, q_logits)

            loss = reconstruction_loss + kl_loss * kl_div_loss_weight

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer.step()
            
            print('[{:4.0f}-{:4.0f} @ {:5.1f} sec] RLoss: {:5.5f} KL Loss: {:1.4f}'
                        .format(
                            epoch,
                            i,
                            time.time() - start,
                            reconstruction_loss.item(),
                            kl_loss.item()
                        ))
main()