'''
    taken from https://nn.labml.ai/sketch_rnn/index.html
'''

import torch
from torch import nn
import einops
from calc_loss import BivariateGaussianMixture


class NNModulePrototype(nn.Module):
    '''
    Parent class to init cuda if available
    Or use enforced device
    '''
    def __init__ (self, enforced_device=None):
        super().__init__()

        if enforced_device is not None:
            self.device = enforced_device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")


class EncoderRNN(NNModulePrototype):
    def __init__(self, d_z: int, enc_hidden_size: int, enforced_device=None):
        super().__init__(enforced_device)
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
    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int, enforced_device=None):
        super().__init__(enforced_device)

        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)
        self.init_state = nn.Linear(d_z, 2*dec_hidden_size)
        self.mixtures = nn.Linear(dec_hidden_size, 6*n_distributions)
        self.q_head = nn.Linear(dec_hidden_size, 3)
        self.q_log_softmax = nn.LogSoftmax(-1)
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size

    def forward(self, x: torch.Tensor, z: torch.Tensor, state = None):
        '''
        state: Optional[Tuple[torch.Tensor, torch.Tensor]]
        '''
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
