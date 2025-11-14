import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append("..")

def getMLP(neurons, activation=nn.GELU, bias=True, dropout=0.1, last_dropout=False, normfun='layernorm'):
    # How to access parameters in module: replace printed < model.0.weight > to < model._modules['0'].weight >
    # neurons: all n+1 dims from input to output
    # len(neurons) = n+1
    # num of params layers = n
    # num of activations = n-1
    if len(neurons) in [0,1]:
        return nn.Identity()
    # if len(neurons) == 2:
    #     return nn.Linear(*neurons)

    nn_list = []
    n = len(neurons)-1
    for i in range(n-1):
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[i+1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[i+1])
        else:
            norm = nn.Identity()
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), norm, activation(), nn.Dropout(dropout)])
    
    nn_list.extend([nn.Linear(neurons[n-1], neurons[n], bias=bias)])
    if last_dropout:
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[-1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[-1])
        else:
            norm = nn.Identity()
        nn_list.extend([norm, activation(), nn.Dropout(dropout)])

    mlp = nn.Sequential(*nn_list)
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    return mlp


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm, alpha):
        super().__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        return x + self.dropout(sublayer(self.norm(x))) #self.alpha: state conservation, 1-self.alpha: state propagation
    
def subsequent_mask(size):
    '''
    mask out subsequent positions.
    param size: int
    return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    bl_k_dft_1 = 1 # 0 is from main diag, 1 is one above
    subsequent_mask = np.triu(np.ones(attn_shape), k=bl_k_dft_1).astype('uint8')
    mask = torch.from_numpy(subsequent_mask) == 0
    return mask # 1 means reachable, 0 means unreachable