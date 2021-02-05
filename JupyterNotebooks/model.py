# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas
import numpy as np

# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    
class RnnEncoder(nn.Module):
    def __init__(self, ent_features, learned_size, n_neurons = 16):
        super(RnnEncoder, self).__init__()

        self.rnn = nn.RNN(input_size = ent_features, hidden_size = learned_size)
        self.learned_size = learned_size
        self.ent_features = ent_features
        
        self.fcn = nn.Sequential(
            nn.Linear(learned_size, 1),
            nn.ReLU()
        )
        
    def init_hidden(self,batch_size):
        self.hidden = torch.zeros(1, batch_size, self.learned_size)
        
    def forward(self, x):
        # input of shape (seq_len, batch, input_size)
        batch_size = len(x)
        
        hiddens = torch.zeros([batch_size, self.learned_size], dtype = torch.float32, device = device)
        outs = torch.zeros([batch_size, 1], dtype = torch.int32, device = device)
        for j in range(batch_size):
            f = x[j]
            entities_count = f.shape[0]
            self.init_hidden(entities_count)
            inp = torch.zeros([entities_count, 1, self.ent_features], dtype = torch.float32, device = device)
            # print(inp.shape, f.shape)
            inp[:,0,:] = f[:,:]
            
            outp, h_n = self.rnn(inp)
            # print('h_N', h_n.shape)
            # print('cell out', outp.shape)
            n = self.fcn(outp)
            
            # https://stackoverflow.com/questions/48005152/extract-the-count-of-positive-and-negative-values-from-an-array
            cou = np.sum(np.array(n.detach().numpy()) > 0, axis = 0)
            # print(cou)
            outs[j] = torch.tensor(cou) 
            hiddens[j] = h_n
            j = j + 1
                
        return outs, hiddens
            
class RnnDecoder(nn.Module):
    def __init__(self, learned_size, dim_features):
        super(RnnDecoder, self).__init__()
        self.rnn = nn.RNN(input_size = learned_size, hidden_size = dim_features)
        self.learned_size = learned_size
        self.dim_features = dim_features
        
    def forward(self, x, out_counts):
        '''Receives tensor with hidden state and number of dimensions tensor per each file.
            Returns a list of tensors with dimension parameters'''
        # input of shape (seq_len, batch, input_size)
        
        batch_size = len(x)
        
        result = []
        for j in range(batch_size):
            dim_count = out_counts[j]
            outs = torch.zeros([dim_count,self.dim_features])
            if dim_count > 0:
                inp = torch.zeros([dim_count, 1, self.learned_size], dtype = torch.float32, device = device)
                xj = x[j]
                # print(inp.shape, xj.shape)
                inp[:,0] = xj
                outs, hiddens = self.rnn(inp)
                print("decoder outs.shape", outs.shape)
                # we remove batch dimension, as batches here are list
                # https://pytorch.org/docs/stable/generated/torch.squeeze.html
                outs = outs.squeeze(dim = 1)
            result.append(outs)
        
        return result