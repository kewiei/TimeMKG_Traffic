import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import EncoderSelfAttn
from .model_utils import *
from .gnn import *

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.num_layers)])
        self.norm = nn.LayerNorm(self.layers[0].size)

    def forward(self, x):
        '''
        param x: src: (batch, N, T_in, C_in)
        return: (batch, N, T_in, C_in)
        '''
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, args, residual_connection=True, use_LayerNorm=True):
        super().__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        size, dropout = args.d_model, args.dropout
        alpha = args.alpha

        self.self_attn = EncoderSelfAttn(args)

        if args.ffn_use_MLP:
            self.ffn = getMLP([args.d_model, int(args.d_model*1.5), args.d_model])
        else:
            self.ffn = spatialGCN_SAt(args, args.norm_Adj_matrix)
        
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout, residual_connection, use_LayerNorm, alpha) for _ in range(2)])
        self.size = size

    def forward(self, x):
        '''
        param x: src: (batch, N, T_in, C_in)
        return: (batch, N, T_in, C_in)
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x, query_multi_segment=True, key_multi_segment=True))
        return self.sublayer[1](x, self.ffn)
    