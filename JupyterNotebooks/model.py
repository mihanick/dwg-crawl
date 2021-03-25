import numpy as np

# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
import torch
from torch import nn

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

class DimRnn(NNModulePrototype):
    def __init__(self, ent_features, dim_features, hidden_size, enforced_device=None):
        '''
        Inints layers and inputs-outputs
        '''
        super().__init__(enforced_device)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=ent_features, hidden_size=self.hidden_size)
        self.ent_features = ent_features
        self.hidden = None
        self.dim_features = dim_features
        self.l1 = nn.Linear(self.hidden_size,self.dim_features)

    def forward(self, x):
        '''

        '''
        # As we have variable entity sequence length between samples in batch
        # we will just manually loop samples in batch
        # while rnn will assume batch_size == 1

        batch_size = len(x)
        
        if self.hidden is None:
            # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            # h_0 of shape (num_layers * num_directions, batch, hidden_size
            self.hidden = torch.zeros(1, 1, self.hidden_size, device=self.device)

        result = torch.zeros(batch_size, self.hidden_size, device=self.device)
        for j in range(batch_size):
            xj = x[j]

            # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            # input of shape (seq_len, batch, input_size)
            inp = xj.unsqueeze(1) 

            # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
            self.hidden = self.hidden.detach()

            # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            # output of shape (seq_len, batch, num_directions * hidden_size)
            # h_n of shape (num_layers * num_directions, batch, hidden_size):
            outp, self.hidden = self.rnn(inp.to(self.device), self.hidden)
            
            result[j] = self.hidden.squeeze(1)
        
        result = self.l1(result)
        # we need an output result shape (batch, dim_features)
        return result

class DimTransformer(NNModulePrototype):
    def __init__(self, ent_features, dim_features, hidden_size, enforced_device=None):
        '''
        Inints layers and inputs-outputs
        '''
        super().__init__(enforced_device)

        self.transformer = torch.nn.Transformer(
            d_model=ent_features, 
            nhead=ent_features,
            dim_feedforward=256,
            num_encoder_layers=4,
            num_decoder_layers=4)

        self.hidden_size = hidden_size
        self.ent_features = ent_features
        self.dim_features = dim_features
        # self.l1 = nn.Linear(self.hidden_size,self.dim_features)

    def forward(self, x, y):
        '''

        '''
        # As we have variable entity sequence length between samples in batch
        # we will just manually loop samples in batch
        # while rnn will assume batch_size == 1

        batch_size = len(x)

        result = torch.zeros(batch_size, self.ent_features, device=self.device)
        for j in range(batch_size):
            xj = x[j]
            yj = y[j]
            # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            # input of shape (seq_len, batch, input_size)
            inp = xj.unsqueeze(1) 
            trgt = yj.unsqueeze(0)
            trgt = trgt.unsqueeze(0)

            # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
            outp = self.transformer(inp.to(self.device), trgt.to(self.device))
            
            r = outp.squeeze(0)
            r = outp.squeeze(0)
            result[j] = r
        # we need an output result shape (batch, dim_features)
        return result.cpu()
