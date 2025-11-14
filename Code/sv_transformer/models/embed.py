import torch
import torch.nn as nn
import math
    
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn = None, smooth_layer_num = 0):

        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to('cuda')  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)

        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)

        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)

        return self.dropout(x)