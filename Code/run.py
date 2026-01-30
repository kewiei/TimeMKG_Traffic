import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_classification import Exp_Classification

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.print_args import print_args
import random
import numpy as np

# def _init_qwen():
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     model_path = "/home/nanodt/gitroot/Qwen3-4B"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=None)
#     return tokenizer, model

edges_pool = ["1-2","1-901","1-903","2-1","2-5","2-905","2-906","5-2","5-6","5-16","6-5","6-8","6-16","6-21","7-5","7-8","8-6","8-7","8-10","9-7","9-10",
                "10-8","10-9","10-12","11-9","11-12","12-10","12-11","12-14","12-23","13-11","13-14","14-12","14-24","14-909","16-6","16-21","16-907",
                "21-6","21-16","21-22","22-21","22-23","22-26","23-12","23-24","23-27","24-27","26-22","26-32","26-34","26-911","27-23","27-28","28-27","28-29","28-32","29-28","29-30",
                "30-29","30-31","30-912","31-29","31-30","31-33","32-26","32-28","32-33","33-31","33-32","33-915","34-26","34-35","35-34","35-36","35-37",
                "36-35","36-37","36-40","37-35","37-38","37-41","38-37",
                "40-36","40-42","41-37","41-40","41-44","42-40","42-45","43-41","43-44","44-41","44-43","44-45","44-46","45-44","45-46","45-917","46-44","46-918","46-919",
                "901-1","903-1","904-1","905-2","907-16","908-13","909-14","910-22","911-26","913-34","914-36","915-33","916-43","917-45","919-46"]

if __name__ == '__main__':
    fix_seed = 1024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimeMKG')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, classification]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='Traffic', help='model id')
    parser.add_argument('--model', type=str, default='TimeMKG8B',
                        help='model name, options: [TimeMKG, TimeMKG8B, Autoformer, TimesNet, iTransformer, DLinear]')

    # data loader  NEW function
    parser.add_argument('--data', type=str, default='Traffic_Multivariate', help='dataset type: options: [Traffic_Singlevariate or Traffic_Multivariate or Traffic_merge],' 
                        'Singlevariate means that it only predicts a single target variable, such as speed, while multivariate means predicting all variables. Traffic_merge can train all links on a single model.')

    parser.add_argument('--root_path', type=str, default='/home/nanodt/gitroot/2V_base/edge_merge_base_large_acc', help='root path of the data file')  #Set according to dataset path
    parser.add_argument('--data_path', type=str, default='id_1-903.csv', help='data file') # single traffic ; Set according to dataset path
    # parser.add_argument('--data_path', type=str, default='merged_traffic.csv', help='data file') # merged traffic ; Set according to dataset path
    
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--target', type=str, default='speed', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--prompt_path', type=str, default='./MKG/traffic.txt', help='location of dataset prompt')

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
    parser.add_argument('--llm_dim', type=int, default=2560, help='data loader num workers. When the model is TimeMKG series, llm_dim will be forced to be 2560 if Qwen-4B is used, 4096 is Qwen-8B is used, as it is the output dimension of different Qwen models.')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data') # Adjust based on GPU memory
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='optimizer learning rate')
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

    # llm_dim will be forced to be 2560 if Qwen-4B is used, 4096 is Qwen-8B is used, as it is the output dimension of different Qwen models
    if args.model == 'TimeMKG':
        print(f'{args.model} is in use, forcing llm_dim to be 2560')
        args.llm_dim = 2560 
    elif args.model == 'TimeMKG8B':
        print(f'{args.model} is in use, forcing llm_dim to be 4096')
        args.llm_dim = 4096
    else:
        pass

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    # # Initialize Qwen model and tokenizer for TimeMKG, so multiple TimeMKG can share the same LLM instance.
    # if args.model == 'TimeMKG':
    #     tokenizer, llm_model = _init_qwen()
    #     args.tokenizer = tokenizer
    #     args.llm_model = llm_model
    #     print('Qwen model and tokenizer loaded.')


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_ei{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.data_path,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.enc_in,
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
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_ei{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.data_path,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.enc_in,
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
            args.des,
            ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         inputs = np.array([[ 10.6818, 83.577 ],
#   [ 10.4175, 392.4576],
#   [ 12.9762, 314.2285],
#   [ 12.2162, 393.4432],
#   [ 11.128 , 259.8624],
#   [ 12.5343, 384.3167],
#   [ 11.336 , 127.5463],
#   [ 13.3389, 315.8902],
#   [  8.916  ,615.9818],
#   [ 11.19  ,  92.0852],
#   [ 10.4097, 387.4082],
#   [ 10.4267, 318.2725]])
        inputs = np.array([[[ 10.6818, 83.577 ],
  [ 10.4175, 392.4576],
  [ 12.9762, 314.2285],
  [ 12.2162, 393.4432],
  [ 11.128 , 259.8624],
  [ 12.5343, 384.3167],
  [ 11.336 , 127.5463],
  [ 13.3389, 315.8902],
  [  8.916  ,615.9818],
  [ 11.19  ,  92.0852],
  [ 10.4097, 387.4082],
  [ 10.4267, 318.2725]]])
        # exp.test(setting, test=1)
        res = exp.predict(setting, inputs)
        print('pred result shape:', res, res.shape)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()