from Code.data_provider.data_factory import data_provider
from Code.exp.exp_basic import Exp_Basic
from Code.utils.tools import EarlyStopping, adjust_learning_rate, visual
from Code.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from Code.utils.dtw_metric import dtw, accelerated_dtw
from Code.utils.augmentation import run_augmentation, run_augmentation_single
from Code.utils.metrics import smape, mase, owa
import configparser
from .connector_1h_class import Connector_1h
warnings.filterwarnings('ignore')

class Old_transformer_Forecast(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self._build_model()
        # super(Old_transformer_Forecast, self).__init__(args)

    def _build_model(self):
        config = configparser.ConfigParser()
        config.read(r'D:\gitroot\TimeMKG_Traffic_full\Code\sv_transformer\flatbushlarge5min.conf')
        self.CT = Connector_1h(config, '0')

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    # CT.predictor_input[step+11,:,0] = pre_input
    # CT.predictor_input[step+11,:,1] = pre_input2
    # CT.read_and_generate_dataset_encoder_decoder(save=False)
    # # all the input has been normalized into range [-1,1] by MaxMin normalization
    # CT._load_graphdata_normY_connector()
    
    # return CT.predict_and_save_results('prediction')
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print('>>> test_data shape:', test_data.data_x.shape, test_data.data_y.shape)
        # self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                print('>>> batch_x shape:', batch_x.shape, 'batch_y shape:', batch_y.shape)
                # batch_x = batch_x.float().to(self.device)
                self.CT.read_and_generate_dataset_encoder_decoder(save=False)
                self.CT._load_graphdata_normY_connector()
                
                outputs = self.CT.predict_and_save_results('prediction')

                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # else:
                #     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 计算naive预测（以真实值的最后一个已知点为naive预测）
        # 假设insample为test_data.insample_y，形状[N, T_insample, D]
        if hasattr(test_data, 'insample_y'):
            insample = test_data.insample_y  # 训练集真实值
        else:
            insample = trues  # 若无训练集，退化为真实值

        naive_preds = np.tile(insample[:, -1:, :], (1, preds.shape[1], 1))  # naive预测

        smape_val = smape(preds, trues)
        mase_val = mase(preds, trues, insample)
        owa_val = owa(preds, trues, insample, naive_preds)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, mape:{}, mspe:{}'.format(mse, mae, mape, mspe))
        print('smape:{}, mase:{}, owa:{}'.format(smape_val, mase_val, owa_val))
        print(smape_val, mase_val, owa_val)
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{}, mspe:{}'.format(mse, mae, mape, mspe))
        f.write('\n')
        f.write('smape:{}, mase:{}, owa:{}'.format(smape_val, mase_val, owa_val))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return