import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

class spatialGCN(nn.Module):
    def __init__(self, args, adj):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = adj  # (N, N)
        self.in_channels = args.d_model
        self.out_channels = args.d_model
        self.Theta = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class spatialGCN_SAt(nn.Module):
    def __init__(self, args, adj):
        super().__init__()
        self.sym_norm_Adj_matrix = adj  # (N, N)
        self.in_channels = args.d_model
        self.out_channels = args.d_model
        self.Theta = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        x = self.dropout(F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)))
        # x.shape: [B,N,T,C]
    
        return x
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)



    def SAt(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))
    

class GCN_PE(nn.Module):    # GCN positional encoding
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super().__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)
    


if __name__ == '__main__':
    # test

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = dotdict({'d_model': 16, 'out_channels': 16, 'dropout': 0.1, 'num_view': 8})
    adj = torch.randn(12, 12)
    x = torch.randn(2, 12, 24, 16)

    