import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_classification import Exp_Classification

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from sv_transformer.exp_old_transformer import Old_transformer_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 1024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimeMKG')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, classification]')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='Traffic', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [TimeMKG, Autoformer, TimesNet, iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, default='Traffic_Multivariate', help='dataset type: options: [Traffic_Singlevariate or Traffic_Multivariate],' 
                        'Singlevariate means that it only predicts a single target variable, such as speed, while multivariate means predicting all variables.')

    # parser.add_argument('--root_path', type=str, default='traffic/edge_merge/', help='root path of the data file')  #Set according to dataset path
    parser.add_argument('--root_path', type=str, default=r'E:/SUMO_Outputs/2V_base/edge_merge/', help='root path of the data file')  #Set according to dataset path
    # parser.add_argument('--data_path', type=str, default='id_1-2.csv', help='data file') # single traffic ; Set according to dataset path
    parser.add_argument('--data_path', type=str, default='merged_traffic.csv', help='data file') # merged traffic ; Set according to dataset path
    
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--target', type=str, default='speed', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--prompt_path', type=str, default='./traffic.txt', help='location of dataset prompt')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length') #input length
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length') # output length
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    # model define
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size, Number of input variables')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size, Number of input variables')

    # parser.add_argument('--c_out', type=int, default=1, help='output size, Number of predictors: Singlevariate') # Singlevariate output
    parser.add_argument('--c_out', type=int, default=2, help='output size, Number of predictors: Singlevariate') # multivariate output

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model') # Model hyperparameters
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads') # Model hyperparameters
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') # Model hyperparameters
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers') # Model hyperparameters
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--max_len', type=int, default=30, help='data loader num workers')
    parser.add_argument('--llm_dim', type=int, default=2560, help='data loader num workers')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data') # Adjust based on GPU memory
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'old_transformer_forecast':
        Exp = Old_transformer_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
