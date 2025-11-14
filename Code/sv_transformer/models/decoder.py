import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import DecoderSelfAttn, DecoderCrossAttn
from .model_utils import *
from .gnn import *

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.num_layers)])
        self.norm = nn.LayerNorm(self.layers[0].size)

    def forward(self, x, memory):
        '''
        param x: (batch, N, T, d_model)
        param memory: (batch, N, T, d_model)
        return: (batch, N, T, d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory)

        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, args, residual_connection=True, use_LayerNorm=True):
        super().__init__()
        self.size = args.d_model

        self.self_attn = DecoderSelfAttn(args)
        self.cross_attn = DecoderCrossAttn(args)
        alpha = args.alpha

        if args.ffn_use_MLP:
            self.ffn = getMLP([args.d_model, int(args.d_model*1.5), args.d_model])
        else:
            self.ffn = spatialGCN_SAt(args, args.norm_Adj_matrix)
        
        
        self.sublayer = nn.ModuleList([SublayerConnection(args.d_model, args.dropout, residual_connection, use_LayerNorm, alpha) for _ in range(3)])


    def forward(self, x, memory):
        '''
        param x: (batch, N, T, C_in)
        param memory: (batch, N, T, C_in)
        return: (batch, N, T, F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device) # (1, T, T)
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T, d_model)
        x = self.sublayer[1](x, lambda x: self.cross_attn(x,m,m, query_multi_segment=False, key_multi_segment=True))  #output: (batch, N, T, d_model)

        return self.sublayer[2](x, self.ffn) #output: (batch, N, T, d_model)
    