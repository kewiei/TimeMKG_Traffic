import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append("..")
from time import time
from .encoder import Encoder
from .decoder import Decoder
from ..libs.utils import norm_Adj
from .embed import TemporalPositionalEncoding, SpatialPositionalEncoding
from .model import CoordinatedCam
from .gnn import GCN_PE



def search_index(max_len, num_of_depend, num_for_predict,points_per_hour, units):
    '''
    Parameters
    ----------
    max_len: int, length of all encoder input
    num_of_depend: int,
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


def make_model(args_out, DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, alpha, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):

    # LR rate means: graph Laplacian Regularization

    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵

    num_of_vertices = norm_Adj_matrix.shape[0]

    # encoder temporal position embedding
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict, num_of_hours * num_for_predict)

    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7*24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index

    print('TemporalPositionalEncoding max_len:', max_len)
    print('w_index:', w_index)
    print('d_index:', d_index)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    args = get_default_args(args_out, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour,
                            num_for_predict, kernel_size, dropout, norm_Adj_matrix, num_layers, decoder_output_size, alpha, aware_temporal_context, ScaledSAt, residual_connection, use_LayerNorm)


    encoder_embedding = nn.Sequential(nn.Linear(encoder_input_size, d_model), TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index), SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN_PE(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num))
    decoder_embedding = nn.Sequential(nn.Linear(decoder_output_size, d_model), TemporalPositionalEncoding(d_model, dropout, num_for_predict), SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN_PE(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num))

    encoder = Encoder(args)

    decoder = Decoder(args)

    model = CoordinatedCam(args,
                      encoder,
                      decoder,
                      encoder_embedding,
                      decoder_embedding,
                      DEVICE)
    # param init
    for p in model.parameters():
        try:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        except ValueError:
            pass


    return model

class D: pass


def get_default_args(args_out, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict, kernel_size, dropout, norm_Adj_matrix, num_layers, decoder_output_size, alpha, aware_temporal_context, ScaledSAt, residual_connection, use_LayerNorm):

    args = D()
    args.args_out = args_out
    # args.adjs = build_non_iso_adjs(args_out)

    args.device                         =           torch.device(f'cuda:{int(args_out["Training"]["cudaID"])}')
    args.num_feat                       =           int(args_out['Training']['in_channels'])
    args.has_PE                         =           False
    args.vis                            =           False

    args.input_embedding_dropout        =           0.05
    args.dim_out                        =           1

    args.transformer_depth              =           2

    args.outer_ffn_dims                 =           []

    model_mode = 'CoordinatedCam'

    # if model_mode == 'CoordinatedCam_notemporal':
    #     # ==== fix below ====
    #     args.ffn_use_MLP        =   1
    #     args.enc_use_graph       =   1
    #     args.use_temporal_aware = 0        
    #     args.use_gcn_sat     =   0
    #     args.dec_use_graph       =   1

    # elif model_mode == 'CoordinatedCam_nograph':
    #     # ==== fix below ====
    #     args.ffn_use_MLP        =   1
    #     args.enc_use_graph       =   0
    #     args.use_temporal_aware = 1
    #     args.use_gcn_sat     =   0
    #     args.dec_use_graph       =   0

    # else:
    #     args.ffn_use_MLP        =   1
    #     args.enc_use_graph       =   1
    #     args.use_temporal_aware = 1 # 0: use multi head attention, 1: use temporal aware attention
    #     args.use_gcn_sat     =   1 # 0: use traditional GCN conv, 1: use spatial attention-based GCN conv
    #     args.dec_use_graph       =   1


    args.ffn_use_MLP        =   1
    args.enc_use_graph       =   1
    args.dec_use_graph       =   1
    args.use_temporal_aware = aware_temporal_context
    args.use_gcn_sat = ScaledSAt 

    args.nb_head            = nb_head
    args.d_model            = d_model
    args.num_for_predict    = num_for_predict
    args.kernel_size        = kernel_size
    args.dropout            = dropout
    args.norm_Adj_matrix    = norm_Adj_matrix
    args.num_layers         = num_layers
    args.decoder_output_size = decoder_output_size
    args.alpha = alpha
    args.K = 3
    args.num_of_weeks = num_of_weeks
    args.num_of_days = num_of_days
    args.num_of_hours = num_of_hours
    args.points_per_hour = points_per_hour
    args.residual_connection = residual_connection
    args.use_LayerNorm = use_LayerNorm

    return args
