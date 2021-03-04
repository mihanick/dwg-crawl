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

class MyRnn(NNModulePrototype):
    def __init__(self, ent_features, dim_features, hidden_size, enforced_device=None):
        super().__init__(enforced_device)

        self.rnn = nn.RNN(input_size=ent_features, hidden_size=hidden_size)
        self.learned_size = hidden_size
        self.ent_features = ent_features
        self.hidden = None
        self.dim_features = dim_features

    def forward(self, x):
        '''

        '''
        batch_size = len(x)
        
        if self.hidden is None:
            self.hidden = torch.zeros(1, batch_size, self.learned_size, device=self.device)

        hiddens = torch.zeros((batch_size, self.learned_size), dtype=torch.float32, device=self.device, requires_grad=True)
        outs = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device, requires_grad=False)

        for j in range(batch_size):
            f = x[j]
            entities_count = f.shape[0]
            inp = f.unsqueeze(1) 

            outp, h_n = self.rnn(inp.to(self.device))
        
        return outp

       