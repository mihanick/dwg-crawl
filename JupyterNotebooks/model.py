'''
    taken from https://nn.labml.ai/sketch_rnn/index.html
'''
import time
import torch
from torch import nn
import einops
from calc_loss import BivariateGaussianMixture, ReconstructionLoss, KLDivLoss
from accuracy import CalculateLoaderAccuracy


class EncoderRNN(nn.Module):
    def __init__(self, d_z: int, enc_hidden_size: int, stroke_features: int):
        super().__init__()
        self.lstm = nn.LSTM(stroke_features, enc_hidden_size, bidirectional=True)
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

class DecoderRNN(nn.Module):
    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int, stroke_features: int):
        super().__init__()

        self.lstm = nn.LSTM(d_z + stroke_features, dec_hidden_size)
        self.init_state = nn.Linear(d_z, 2*dec_hidden_size)
        self.mixtures = nn.Linear(dec_hidden_size, 6*n_distributions)
        self.q_head = nn.Linear(dec_hidden_size, stroke_features - 2) #first two stroke features are dx,dy than follow logits
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

class Trainer:
    def __init__(self, dwg_dataset, lr=0.008, train_verbose=True):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.train_verbose = train_verbose

        self.dwg_dataset = dwg_dataset

        self.train_loader = dwg_dataset.train_loader
        self.val_loader   = dwg_dataset.val_loader
        self.test_loader  = dwg_dataset.test_loader

        self.stroke_features = dwg_dataset.entities.stroke_features
        self.max_seq_length = dwg_dataset.entities.max_seq_length

        self.enc_hidden_size    = 256
        self.dec_hidden_size    = 512

        self.d_z                = 128
        self.n_distributions    = 20
        self.kl_div_loss_weight = 0.8
        self.grad_clip          = 1.0
        self.temperature        = 0.4

        self.encoder = EncoderRNN(self.d_z, self.enc_hidden_size, stroke_features=self.stroke_features)
        self.decoder = DecoderRNN(self.d_z, self.dec_hidden_size, self.n_distributions, stroke_features=self.stroke_features)
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)

    def train_epoch(self, epoch_no):
        start = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.encoder.train()
        self.decoder.train()

        for i, batch in enumerate(self.train_loader):
            data = batch[0].to(self.device).transpose(0, 1)
            mask = batch[1].to(self.device).transpose(0, 1)

            z, mu, sigma_hat = self.encoder(data)

            z_stack = z.unsqueeze(0).expand(data.shape[0]- 1, -1, -1)

            inputs = torch.cat([data[:-1], z_stack], 2)

            dist, q_logits, _ = self.decoder(inputs, z, None)

            test_kl = KLDivLoss()(sigma_hat, mu)
            test_rl = ReconstructionLoss()(mask, data[1:], dist, q_logits)

            loss = test_rl + test_kl * self.kl_div_loss_weight

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

            self.optimizer.step()
            if self.train_verbose:
                print('  [{:4.0f}-{:4.0f} @ {:5.1f} sec] RLoss: {:5.5f} KL Loss: {:1.4f}'
                            .format(
                                epoch_no,
                                i,
                                time.time() - start,
                                test_rl.item(),
                                test_kl.item()
                            ))
        
        # validation
        val_kl, val_rl = CalculateLoaderAccuracy(self.encoder, self.decoder, self.val_loader, self.device)

        if self.train_verbose:
            print('Epoch [{} @ {:4.1f}] validation losses rl:{:1.4f} kl:{:1.4f}'.format(epoch_no, time.time() - start, val_rl, val_kl))

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save model
        torch.save(self.encoder.state_dict(), 'DimEncoder.model')
        torch.save(self.decoder.state_dict(), 'DimDecoder.model')

        return test_rl, test_kl, val_rl, val_kl
