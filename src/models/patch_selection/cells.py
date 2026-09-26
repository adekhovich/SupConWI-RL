import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# !!!!  https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

class MyLSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def apply_timestep(self, x, hidden):
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, F.tanh(cy))
        
        return (hy, cy)
    
    def forward(self, x, hidden, t=0):
        
        hidden = self.apply_timestep(x, hidden) 
        
        return hidden
    
    

class MyGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0, bias=True):
        super(MyGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias
        self.x2h = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        #self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def apply_timestep(self, x, hidden):
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        #print("@", gate_x.size(), gate_h.size())
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        #print(i_r.size(), h_r.size())
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + (reset_gate * h_n))
        
        hy = new_gate + input_gate * (hidden - new_gate)
        
        return hy
        
    
    def forward(self, x, hidden, t=0):
        
        hidden = self.apply_timestep(x, hidden) 
        
        return hidden