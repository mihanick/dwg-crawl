import numpy as np

# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
import torch
from torch import nn

class RnnEncoder(nn.Module):
    def __init__(self, ent_features, learned_size, enforced_device=None):
        super(RnnEncoder, self).__init__()

        self.rnn = nn.RNN(input_size=ent_features, hidden_size=learned_size)
        self.learned_size = learned_size
        self.ent_features = ent_features
        self.hidden = None

        self.fcn1 = nn.Linear(learned_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fcn2 = nn.Linear(256, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fcno = nn.Sequential(
            nn.Linear(1024, 1),
            nn.ReLU()
        )
        
        if enforced_device is not None:
            self.device = enforced_device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        
    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.learned_size, device=self.device)
        
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
            oo = self.fcn1(outp)
            oo = oo.squeeze(dim=1)
            oo = self.bn1(oo)
            oo = self.fcn2(oo)
            oo = self.bn2(oo)
            oo = self.fcno(oo)
            
            # https://stackoverflow.com/questions/48005152/extract-the-count-of-positive-and-negative-values-from-an-array
            _count = np.sum(np.array(oo.cpu().detach().numpy()) > 0, axis=None)
            
            outs[j] = _count
            hiddens[j] = h_n
        
        return outs, hiddens
            
class RnnDecoder(nn.Module):
    def __init__(self, learned_size, dim_features, enforced_device=None):
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

class Padder(NNModulePrototype):
    def __init__(self, max_seq_length, enforced_device=None, requires_grad=False):
        super().__init__(enforced_device)

        self.max_seq_length = max_seq_length
        self.requires_grad = requires_grad

    def forward(self, x):
        '''
        takes input of shape List(seq_len, ent_features).len=batch_size
        and outputs it as 0-padded tensor(batchб max_seq_length, ent_features)
        '''
        batch_size = len(x)
        ent_features = x[0].shape[1]
        output = torch.zeros((batch_size, self.max_seq_length, ent_features), dtype=torch.float32, device=self.device, requires_grad=self.requires_grad)

        i = 0
        for xx in x:
            assert xx.shape[0] <= self.max_seq_length, "Input length should be less than max sequence length"
            # https://stackoverflow.com/questions/48686945/reshaping-a-tensor-with-padding-in-pytorch
            output[i] = torch.nn.functional.pad(xx, (0, 0, 0, self.max_seq_length - xx.shape[0]), mode='constant', value=0)
            i = i + 1

        return output


class UglyModel(NNModulePrototype):
    def __init__ (self, ent_features, dim_features, max_seq_length, enforced_device=None):
        super(UglyModel, self).__init__(enforced_device=enforced_device)
        
        self.ent_features = ent_features
        self.dim_features = dim_features
        self.max_seq_length =max_seq_length

        self.hidden_size = self.dim_features*20

        self.output_seq_length =  int(max_seq_length/100)

        self.fc0 = nn.Linear(self.max_seq_length, self.hidden_size)
        
        self.fc1 = nn.Linear(self.hidden_size * self.ent_features, self.dim_features * self.output_seq_length)
        self.relu = nn.ReLU()

        # this requires dropLast for dataset, as we can have 1- length batch
        #https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/5
        self.bn1 = nn.BatchNorm1d(self.dim_features * self.output_seq_length)

        self.do = nn.Dropout(0.95)

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        x = torch.transpose(x, 1, 2)
        x = self.fc0(x)
        x = x.view(batch_size, self.hidden_size * self.ent_features)
        
        x = self.fc1(x)
        x = self.bn1(x)
        
        x = self.relu(x)
        

        # x = self.do(x)
        x = x.view(batch_size, self.dim_features, self.output_seq_length)
        x = torch.transpose(x, 1, 2)
        
        return x