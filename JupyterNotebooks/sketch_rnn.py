
'''
Taken from https://github.com/alexis-jacq/Pytorch-Sketch-RNN/blob/master/sketch_rnn.py
'''
import numpy as np
import matplotlib.pyplot as plt
import PIL
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

torch.set_printoptions(profile='short')

def pretty_print(*print_args):
    torch.set_printoptions(precision=3, profile="full", sci_mode=False)
    print(print_args)
    torch.set_printoptions(profile="short")

class EncoderRNN(nn.Module):
    def __init__(self, Nz, enc_hidden_size, dropout, stroke_features, device):
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.stroke_features = stroke_features
        self.device = device

        self.lstm = nn.LSTM(stroke_features, self.enc_hidden_size, dropout=dropout, bidirectional=True)

        #mu and sigma layers from lstm last output
        self.fc_mu = nn.Linear(2 * self.enc_hidden_size, Nz)
        self.fc_sigma = nn.Linear(2 * self.enc_hidden_size, Nz)

    def forward(self, inputs, hidden_cell=None):
        batch_size = inputs.shape[1]
        if hidden_cell is None:
            #init with zeroes
            hidden = torch.zeros(2, batch_size, self.enc_hidden_size, device=self.device)
            cell = torch.zeros(2, batch_size, self.enc_hidden_size, device=self.device)
        
            hidden_cell = (hidden, cell)

        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size) we need -> (batch_size, 2*hidden_size)
        
        hidden_fw, hiiden_bw = torch.split(hidden, 1, 0)
        hidden_catted = torch.cat([hidden_fw.squeeze(0), hiiden_bw.squeeze(0)],1)

        # mu & sigma
        mu = self.fc_mu(hidden_catted)
        sigma_hat = self.fc_sigma(hidden_catted)
        sigma = torch.exp(sigma_hat / 2)

        N = torch.normal(torch.zeros_like(mu), torch.ones_like(mu)).to(self.device)

        z = mu + sigma * N

        # mu and sigma for LKL loss
        return z, mu, sigma_hat

class DecoderRNN(nn.Module):
    def __init__(self, Nz, M_size, stroke_features, max_seq_length, dec_hidden_size, dropout, device):
        super().__init__()
        self.dec_hidden_size = dec_hidden_size
        self.M = M_size
        self.device = device
        self.max_seq_length = max_seq_length
        self.stroke_features = stroke_features

        # init hidden and cell from z
        self.fc_hc = nn.Linear(Nz, 2 * dec_hidden_size)

        # unidirectional lstm
        self.lstm = nn.LSTM(Nz + stroke_features, dec_hidden_size, dropout=dropout)

        # probability distribution parameters from hiddens
        logits_no = stroke_features - 2
        self.fc_params = nn.Linear(self.dec_hidden_size, 2 * logits_no * self.M + logits_no)
        self.dropout_fc_params = nn.Dropout()

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # init form z
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        
        # wtf why?
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed the last generated sample
        if self.training:
            y = self.fc_params(outputs.view(-1, self.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.dec_hidden_size))

        # separate pen and mixture params
        # 6 is for pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy
        params = torch.split(y, 6, 1)
        # trajectory
        params_mixture = torch.stack(params[:-1])
        # pen up/down
        params_pen = params[-1]
        # print(params_pen)

        # identify mixture parameters:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        # preprocess params:
        len_out = 1
        if self.training:
            len_out = self.max_seq_length

        num_logits = self.stroke_features - 2

        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=0).view(len_out, -1, self.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, self.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, self.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, self.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.M)
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, num_logits)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

class Trainer:
    def __init__(self, dwg_dataset, lr=0.001, train_verbose=True):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.enc_hidden_size = 256
        self.dec_hidden_size = 512

        self.Nz              = 128
        self.M               = 20

        self.dropout         = 0.9
        
        #self.eta_min     = 0.01
        #self.R           = 0.99995
        self.KL_min        = 1.001
        self.min_reconstruction_loss = 0.0001
        self.wKL           = 0.5
        self.lr            = lr
        self.lr_decay      = 0.98
        self.min_lr        = 1e-6
        self.grad_clip     = 1.0
        self.temperature   = 0.5

        self.train_verbose   = train_verbose
        self.dwg_dataset     = dwg_dataset

        self.train_loader    = dwg_dataset.train_loader
        self.val_loader      = dwg_dataset.val_loader
        self.test_loader     = dwg_dataset.test_loader

        self.stroke_features = dwg_dataset.entities.stroke_features
        self.max_seq_length  = dwg_dataset.entities.max_seq_length
        self.batch_size      = dwg_dataset.batch_size

        self.enc_hidden_size = 256
        self.dec_hidden_size = 512

        self.Nz              = 32
        self.M               = 20

        self.encoder = EncoderRNN(
            Nz=self.Nz,
            enc_hidden_size=self.enc_hidden_size, 
            stroke_features=self.stroke_features, 
            dropout=self.dropout,
            device=self.device)
        self.decoder = DecoderRNN(
            Nz = self.Nz,
            dec_hidden_size=self.dec_hidden_size, 
            M_size=self.M,
            max_seq_length=self.max_seq_length,
            dropout=self.dropout, 
            stroke_features=self.stroke_features, 
            device=self.device)
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

    def adjust_lr_decay(self, optimizer):
        'Learning rate decay'
        for param_group in optimizer.param_groups:
            if param_group['lr']> self.min_lr:
                param_group['lr'] *= self.lr_decay
        return optimizer

    def make_target(self, batch):
        '''
        batch[max_seq_length,batch_size,stroke_features]
        '''

        mask = 1 - batch[:, :, 4]

        dx = torch.stack([batch.data[:, :, 0]] * self.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * self.M, 2)
        p1 = batch.data[:, :, 2]
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p

    def new_stroke_sample(self, pi, q, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        def adjust_temperature(pi_pdf):
            pi_pdf = np.log(pi_pdf) / self.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature, greedy=False):
            if greedy:
                return mu_x, mu_y
            mean = [mu_x, mu_y]

            sigma_x *= np.sqrt(temperature)
            sigma_y *= np.sqrt(temperature)

            cov = [[sigma_x*sigma_x, rho_xy*sigma_x*sigma_y], [rho_xy*sigma_x*sigma_y, sigma_y*sigma_y]]
            
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        #get mixture index
        pi = adjust_temperature(pi)
        pi_idx = np.random.choice(self.M, p=pi)

        #get pen state
        q = adjust_temperature(q)
        # TODO: There could be only one output for logits
        # where q as probability distribution of logits
        # q_idx is selected on base of probablities
        # where should be 1 in data
        q_idx = np.random.choice(self.stroke_features - 2, p=q)
        logits = torch.zeros(self.stroke_features - 2)
        logits[q_idx] = 1

        #get mixture params:
        mu_x = mu_x.data[pi_idx].cpu().numpy()
        mu_y = mu_y.data[pi_idx].cpu().numpy()
        sigma_x = sigma_x.data[pi_idx].cpu().numpy()
        sigma_y = sigma_y.data[pi_idx].cpu().numpy()
        rho_xy = rho_xy.data[pi_idx].cpu().numpy()

        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, self.temperature, greedy=False)

        result = torch.FloatTensor([x, y])
        result = torch.cat([result, logits])
        return result
            
    def sample_next_state(self, pi, q, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        '''
        pi.shape[1,batch_size,M]
        q.shape[1,batch_size,stroke_features - 2(for x,y) = 3]
        mu_x,mu_y,sigma_x,sigma_y,rho_xy.shape[1,batch_size, M]
        returns next sketch shape[batch_size, stroke_features]
        '''

        with torch.no_grad():
            result = torch.zeros(self.batch_size, self.stroke_features)

            for i in range(self.batch_size):
                #dx, dy, is_pen, is_lift, eos
                result[i] = self.new_stroke_sample(
                    pi[0, i].cpu().numpy(),
                    q[0, i].cpu().numpy(),
                    mu_x[0, i],
                    mu_y[0, i],
                    sigma_x[0, i],
                    sigma_y[0, i],
                    rho_xy[0, i]
                    )

            return result

    def reconstruction_loss(self, mask, dx, dy, p, mu_x, mu_y, sigma_x, sigma_y, rho_xy, pi, q):
        '''
            calculates losses of stroke (dx, dy) and pen (p1, p2, p3)
        '''
        def bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
            z_x = ((dx - mu_x) / sigma_x) ** 2
            z_y = ((dy - mu_y) / sigma_y) ** 2
            z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
            z = z_x + z_y - 2 * rho_xy * z_xy
            exp_ = torch.exp(-z/(2 * (1 - rho_xy ** 2)))
            norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)

            result = exp_ / norm

            # when rho_xy==1 result contains nan 
            # and it's baAad
            # torch 1.8.1
            # result = torch.nan_to_num(result, nan=0.999994)
            result[torch.isnan(result)] = 0.999995

            return result

        pdf = bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        loss_stroke = -torch.sum(mask * torch.log(1e-5 + torch.sum(pi * pdf, 2)))
        loss_pen = -torch.sum(p * torch.log(q))

        loss_pen /= self.max_seq_length
        loss_stroke /= self.max_seq_length

        return loss_stroke, loss_pen

    def kullback_leibler_loss(self, sigma, mu):
        '''
        Gotta wiki this. 
        As far I understand, that's some difference between 
        stochastic distributions
        '''
        LKL = -0.5 * torch.sum(1 + sigma - mu**2 - torch.exp(sigma)) / float(self.Nz * self.batch_size)
        # KL_min = torch.Tensor([self.KL_min]).to(self.device)
        # LKL =  torch.max(LKL, KL_min) # * eta_step
        if torch.isnan(LKL):
            LKR = self.KL_min
        LKL *= self.wKL 
        return LKL

    def CalculateLoaderAccuracy(self, loader):
        self.encoder.eval()
        self.decoder.eval()

        ls = []
        lp = []
        kl = []
        with torch.no_grad():
            for i, b in enumerate(loader):
                batch = b[0].to(self.device).transpose(0, 1)
                loss_kl, ls_, lp_ = self.loss_on_batch(batch)
                ls.append(float(ls_))
                lp.append(float(lp_))
                kl.append(float(loss_kl))

            return np.mean(ls), np.mean(lp), np.mean(kl)

    def loss_on_batch(self, batch):
        '''
        forwards batch through decoder and encoder
        and calculates losses

        returns
        ls - loss stroke
        lp - loss pen
        lkl - loss kullback leibler
        '''
        z, mu, sigma = self.encoder(batch)

        # expand z in order it to be able to concatenate with inputs
        z_stack = torch.stack([z] * (self.max_seq_length))

        # inputs is the concatination of z and batch_inputs:
        inputs = torch.cat([batch, z_stack], 2)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.decoder(inputs, z)

        # prepare targets:
        mask, dx, dy, p = self.make_target(batch)

        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss_kl = self.kullback_leibler_loss(sigma, mu)
        loss_sketch, loss_pen = self.reconstruction_loss(mask, dx, dy, p, mu_x, mu_y, sigma_x, sigma_y, rho_xy, pi, q)

        return loss_sketch, loss_pen, loss_kl

    def train_epoch(self, epoch_no):
        start = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.encoder.train()
        self.decoder.train()

        train_ls_losses =  []
        train_lp_losses =  []
        train_kl_losses = []
        loss = 0
        for i, batch in enumerate(self.train_loader):
            # TODO: make this transpose in dataset
            batch = batch[0].to(self.device).transpose(0, 1)

            loss_stroke, loss_pen, loss_kl = self.loss_on_batch(batch)

            loss = loss_kl + loss_stroke + loss_pen
            loss.backward()

            # grad clip
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()
            
            self.enc_optimizer = self.adjust_lr_decay(self.enc_optimizer)
            self.dec_optimizer = self.adjust_lr_decay(self.dec_optimizer)

            train_ls_losses.append(float(loss_stroke))
            train_lp_losses.append(float(loss_pen))
            train_kl_losses.append(float(loss_kl))

        # train
        train_ls = np.mean(train_ls_losses)
        train_lp = np.mean(train_lp_losses)
        train_kl = np.mean(train_kl_losses)
        # validation
        val_ls, val_lp, val_kl = self.CalculateLoaderAccuracy(self.val_loader)

        print('Epoch [{} @ {:4.1f}] train losses: ls:{:1.4f} lp:{:1.4f} kl:{:1.4f} validation losses ls:{:1.4f} lp:{:1.4f} kl:{:1.4f}'.format(
            epoch_no,
            time.time() - start,
            train_ls,
            train_lp,
            train_kl,
            val_ls,
            val_lp,
            val_kl))

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save model
        if epoch_no % 10 == 0:
            torch.save(self.encoder.state_dict(), 'DimEncoder.model')
            torch.save(self.decoder.state_dict(), 'DimDecoder.model')

        return train_ls, train_lp, train_kl, val_ls, val_lp, val_kl
