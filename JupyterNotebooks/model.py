import numpy as np

# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
import torch
from torch import nn

class RnnEncoder(nn.Module):
    def __init__(self, ent_features, learned_size, enforced_device=None):
        super(RnnEncoder, self).__init__()

        self.rnn = nn.RNN(input_size = ent_features, hidden_size = learned_size)
        self.learned_size = learned_size
        self.ent_features = ent_features
        self.hidden = None

        self.fcn = nn.Sequential(
            nn.Linear(learned_size, 128),
            # nn.Dropout(0.51),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        
        if enforced_device is not None:
            self.device = enforced_device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        
    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.learned_size, device = self.device)
        
    def forward(self, x):
        # input of shape (seq_len, batch, input_size)
        batch_size = len(x)
        
        hiddens = torch.zeros((batch_size, self.learned_size), dtype=torch.float32, device=self.device, requires_grad=True)
        outs = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device, requires_grad=False)

        for j in range(batch_size):
            f = x[j]
            entities_count = f.shape[0]
            self.init_hidden(entities_count)
            inp = f.unsqueeze(1) 

            # print(inp.shape, f.shape)
            outp, h_n = self.rnn(inp.to(self.device))

            # print('h_N', h_n.shape)
            # print('cell out', outp.shape)
            n = self.fcn(outp)
            
            # https://stackoverflow.com/questions/48005152/extract-the-count-of-positive-and-negative-values-from-an-array
            _count = np.sum(np.array(n.cpu().detach().numpy()) > 0, axis=None)
            
            outs[j] = _count
            hiddens[j] = h_n
        
        return outs, hiddens
            
class RnnDecoder(nn.Module):
    def __init__(self, learned_size, dim_features, enforced_device = None):
        super(RnnDecoder, self).__init__()
        self.rnn = nn.RNN(input_size=learned_size, hidden_size=dim_features)
        self.learned_size = learned_size
        self.dim_features = dim_features
        
        if enforced_device is not None:
            self.device = enforced_device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                
    def forward(self, out_counts, x):
        '''Receives tensor with hidden state and number of dimensions tensor per each file.
            Returns a list of tensors with dimension parameters'''
        
        batch_size = len(x)
        
        result = []
        for j in range(batch_size):
            dim_count = int(out_counts[j].item())
            outs = torch.zeros((dim_count, self.dim_features), dtype=torch.float32, device=self.device, requires_grad=True)
            if dim_count > 0:
                inp = torch.zeros((dim_count, self.learned_size), dtype=torch.float, device=self.device)
                # that is not what supposed to happen, as we are just copying inputs to each dimension output
                inp[:, :] = x[j, :]
                inp = inp.unsqueeze(1)
                # for nn.rnn  input of shape (seq_len, batch, input_size)
                outs, _hidden = self.rnn(inp)
                # print("decoder outs.shape", outs.shape)
                # we remove batch dimension, as batches here are represented as list
                # https://pytorch.org/docs/stable/generated/torch.squeeze.html
                outs = outs.squeeze(dim=1)
            result.append(outs)
        
        return result