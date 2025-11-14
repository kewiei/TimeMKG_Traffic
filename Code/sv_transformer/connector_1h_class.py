#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from .models.functions import make_model
from .libs.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, masked_mape_np, load_graphdata_normY_connector
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from tensorboardX import SummaryWriter


class Connector_1h:
    def __init__(self, config, cuda):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        USE_CUDA = torch.cuda.is_available()
        self.DEVICE = torch.device('cuda:0')
        print("CUDA:", USE_CUDA, self.DEVICE, flush=True)

        data_config = config['Data']
        training_config = config['Training']
        connector_config = config['Connector']

        # check if multiple GPUs are available. If not, use default GPU: 0
        if int(training_config["multiGPU"]):
            os.environ['CUDA_VISIBLE_DEVICES'] = training_config["whichGPUs"]
            self.DEVICE = torch.device(f'cuda:0')
            training_config["DEVICE_ID"] = "0"
        else:
            self.DEVICE = torch.device(f'cuda:{int(training_config["cudaID"])}')
            training_config["DEVICE_ID"] = training_config["cudaID"]
            
        self.adj_filename = data_config['adj_filename']

        self.volume_matrix_filename = connector_config['volume_matrix_filename']
        self.speed_matrix_filename = connector_config['speed_matrix_filename']

        if config.has_option('Data', 'id_filename'):
            self.id_filename = data_config['id_filename']
        else:
            self.id_filename = None
        self.num_of_vertices = int(data_config['num_of_vertices'])
        self.points_per_hour = int(data_config['points_per_hour'])
        self.num_for_predict = int(data_config['num_for_predict'])
        self.dataset_name = data_config['dataset_name']
        self.model_name = training_config['model_name']
        self.learning_rate = float(training_config['learning_rate'])
        self.best_epoch = int(connector_config['best_epoch'])
        print('best_epoch:', self.best_epoch, flush=True)
        self.batch_size = int(training_config['batch_size'])
        print('batch_size:', self.batch_size, flush=True)
        self.num_of_weeks = int(training_config['num_of_weeks'])
        self.num_of_days = int(training_config['num_of_days'])
        self.num_of_hours = int(training_config['num_of_hours'])
        self.direction = int(training_config['direction'])
        self.encoder_input_size = int(training_config['encoder_input_size'])
        self.decoder_input_size = int(training_config['decoder_input_size'])
        self.dropout = float(training_config['dropout'])
        self.kernel_size = int(training_config['kernel_size'])

        self.num_layers = int(training_config['num_layers'])
        self.d_model = int(training_config['d_model'])
        self.nb_head = int(training_config['nb_head'])
        self.ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
        self.SE = bool(int(training_config['SE']))  # whether use spatial embedding
        self.smooth_layer_num = int(training_config['smooth_layer_num'])
        self.aware_temporal_context = bool(int(training_config['aware_temporal_context']))
        self.TE = bool(int(training_config['TE']))
        self.use_LayerNorm = True
        self.residual_connection = True

        self.alpha = float(training_config['alpha'])
        self.graph_edge_filename = connector_config['graph_edge_filename']

        # self.direction = 1 means: if i connected to j, adj[i,j]=1;
        # self.direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
        if self.direction == 2:
            adj_mx, distance_mx = get_adjacency_matrix_2direction(self.adj_filename, self.num_of_vertices, self.id_filename)
        if self.direction == 1:
            adj_mx, distance_mx = get_adjacency_matrix(self.adj_filename, self.num_of_vertices, self.id_filename)
        folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (self.model_name, self.num_of_hours, self.num_of_days, self.num_of_weeks, self.num_layers, self.nb_head, self.d_model, self.encoder_input_size, self.direction, self.dropout, self.learning_rate)

        if self.aware_temporal_context:
            folder_dir = folder_dir+'Tcontext'
        if self.ScaledSAt:
            folder_dir = folder_dir + 'ScaledSAt'
        if self.SE:
            folder_dir = folder_dir + 'SE' + str(self.smooth_layer_num)
        if self.TE:
            folder_dir = folder_dir + 'TE'

        print('folder_dir:', folder_dir, flush=True)
        self.params_path = os.path.join('D:\gitroot\DT-DIMA-EM\experiments', self.dataset_name, folder_dir)

        self.predictor_input = self.get_graph_signal_matrix()

        self.all_data = self.read_and_generate_dataset_encoder_decoder(save=False)

        # all the input has been normalized into range [-1,1] by MaxMin normalization
        self._load_graphdata_normY_connector()
            
        self.net = make_model(config, self.DEVICE, self.num_layers, self.encoder_input_size, self.decoder_input_size, self.d_model, adj_mx, self.nb_head, self.num_of_weeks,
                        self.num_of_days, self.num_of_hours, self.points_per_hour, self.num_for_predict, alpha=self.alpha, dropout=self.dropout, aware_temporal_context=self.aware_temporal_context, 
                        ScaledSAt=self.ScaledSAt, SE=self.SE, TE=self.TE, kernel_size=self.kernel_size, smooth_layer_num=self.smooth_layer_num, residual_connection=self.residual_connection, 
                        use_LayerNorm=self.use_LayerNorm)

        print(self.net, flush=True)
    
    def _load_graphdata_normY_connector(self):
        self.test_loader, self.data_target_tensor, self._max, self._min = load_graphdata_normY_connector(
            self.all_data, self.num_of_hours,
            self.num_of_days, self.num_of_weeks, self.DEVICE, self.batch_size)


    def search_data(self, sequence_length, num_of_depend, label_start_idx, units, ):
        '''
        Parameters
        ----------
        sequence_length: int, length of all history data
        num_of_depend: int,
        label_start_idx: int, the first index of predicting target
        self.num_for_predict: int, the number of points will be predicted for each sample
        units: int, week: 7 * 24, day: 24, recent(hour): 1
        self.points_per_hour: int, number of points per hour, depends on data
        Returns
        ----------
        list[(start_idx, end_idx)]
        '''

        if self.points_per_hour < 0:
            raise ValueError("self.points_per_hour should be greater than 0!")

        if label_start_idx + self.num_for_predict > sequence_length:
            return None

        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + self.num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None

        if len(x_idx) != num_of_depend:
            return None

        return x_idx[::-1]

    def get_sample_indices(self, data_sequence, label_start_idx):
        '''
        Parameters
        ----------
        data_sequence: np.ndarray
                    shape is (sequence_length, self.num_of_vertices, num_of_features)
        self.num_of_weeks, self.num_of_days, self.num_of_hours: int
        label_start_idx: int, the first index of predicting target, 预测值开始的那个点
        self.num_for_predict: int,
                        the number of points will be predicted for each sample
        self.points_per_hour: int, default 12, number of points per hour
        Returns
        ----------
        week_sample: np.ndarray
                    shape is (self.num_of_weeks * self.points_per_hour,
                            self.num_of_vertices, num_of_features)
        day_sample: np.ndarray
                    shape is (self.num_of_days * self.points_per_hour,
                            self.num_of_vertices, num_of_features)
        hour_sample: np.ndarray
                    shape is (self.num_of_hours * self.points_per_hour,
                            self.num_of_vertices, num_of_features)
        target: np.ndarray
                shape is (self.num_for_predict, self.num_of_vertices, num_of_features)
        
        '''
        week_sample, day_sample, hour_sample = None, None, None

        if label_start_idx + self.num_for_predict > data_sequence.shape[0]:
            return week_sample, day_sample, hour_sample, None

        if self.num_of_weeks > 0:
            week_indices = self.search_data(data_sequence.shape[0], self.num_of_weeks,
                                    label_start_idx, 7 * 24,)
            if not week_indices:
                return None, None, None, None

            week_sample = np.concatenate([data_sequence[i: j]
                                        for i, j in week_indices], axis=0)

        if self.num_of_days > 0:
            day_indices = self.search_data(data_sequence.shape[0], self.num_of_days,
                                    label_start_idx, 24,)
            if not day_indices:
                return None, None, None, None

            day_sample = np.concatenate([data_sequence[i: j]
                                        for i, j in day_indices], axis=0)

        if self.num_of_hours > 0:
            hour_indices = self.search_data(data_sequence.shape[0], self.num_of_hours,
                                    label_start_idx, 1,)
            if not hour_indices:
                return None, None, None, None

            hour_sample = np.concatenate([data_sequence[i: j]
                                        for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + self.num_for_predict]
            
        return week_sample, day_sample, hour_sample, target

    def get_graph_signal_matrix(self):

        import pandas as pd
        
        data_list_vol = pd.read_csv(self.volume_matrix_filename)
        data_list_vol = data_list_vol.groupby(data_list_vol.index//5).mean()
        #ffill here
        data_list_vol = data_list_vol.ffill()
        data_list_vol = data_list_vol.fillna(1)

        data_list_spd = pd.read_csv(self.speed_matrix_filename)
        data_list_spd = data_list_spd.groupby(data_list_spd.index//5).mean()
        #ffill here
        data_list_spd = data_list_spd.ffill()
        data_list_spd = data_list_spd.fillna(0)

        data_seq_vol = data_list_vol.iloc[:,1:].to_numpy()
        data_seq_vol = np.expand_dims(data_seq_vol, axis=2)

        data_seq_spd = data_list_spd.iloc[:,1:].to_numpy()
        data_seq_spd = np.expand_dims(data_seq_spd, axis=2)
        data_seq = np.concatenate((data_seq_vol, data_seq_spd), axis=2)

        return data_seq

    def MinMaxnormalization(self, test):
        '''
        Parameters
        ----------
        test: np.ndarray (B,N,F,T)
        Returns
        ----------
        stats: dict, two keys: mean and std
        test_norm: np.ndarray,
                                        shape is the same as original
        '''

        self._max = test.max(axis=(0, 1, 3), keepdims=True)
        self._min = test.min(axis=(0, 1, 3), keepdims=True)

        print('_max.shape:', self._max.shape)
        print('_min.shape:', self._min.shape)

        def normalize(x):
            x = 1. * (x - self._min) / (self._max - self._min)
            x = 2. * x - 1.
            return x

        test_norm = normalize(test)

        return {'_max': self._max, '_min': self._min}, test_norm

    def re_max_min_normalization(self, x,):
        x = (x + 1.) / 2.
        x = 1. * x * (self._max - self._min) + self._min
        return x

    def read_and_generate_dataset_encoder_decoder(self, save=False):
        '''
        Parameters
        ----------
        graph_signal_matrix_filename: str, path of graph signal matrix file
        self.num_of_weeks, self.num_of_days, self.num_of_hours: int
        self.num_for_predict: int
        self.points_per_hour: int, default 12, depends on data

        Returns
        ----------
        feature: np.ndarray,
                shape is (num_of_samples, num_of_depend * self.points_per_hour,
                        self.num_of_vertices, num_of_features)
        target: np.ndarray,
                shape is (num_of_samples, self.num_of_vertices, self.num_for_predict)
        
        '''
        data_seq = self.predictor_input # (sequence_length, self.num_of_vertices, num_of_features)

        all_samples = []
        for idx in range(data_seq.shape[0]):
            sample = self.get_sample_indices(data_seq, idx)
            if ((sample[0] is None) and (sample[1] is None) and (sample[3] is None)):
                continue

            week_sample, day_sample, hour_sample, target = sample

            sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample,observation]

            if self.num_of_weeks > 0:
                week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(week_sample)

            if self.num_of_days > 0:
                day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(day_sample)

            if self.num_of_hours > 0:
                hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, :, :]  # (1,N,T)
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(
                sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1), (1,N,Tobs)]

        testing_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples)]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1),(B,N,Tobs)]

        test_x = np.concatenate(testing_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together

        test_target = testing_set[-2] # (B,N,F,T)

        test_timestamp = testing_set[-1] # (B,1)
        

        # max-min normalization on x
        (stats, test_x_norm) = self.MinMaxnormalization(test_x)

        self.all_data = {
            'test': {
                'x': test_x_norm,
                'target': test_target,
                'timestamp': test_timestamp,
            },
            'stats': {
                '_max': stats['_max'],
                '_min': stats['_min'],
            }
        }
        
        print('test x:', self.all_data['test']['x'].shape)
        print('test target:', self.all_data['test']['target'].shape)
        print('test timestamp:', self.all_data['test']['timestamp'].shape)
        print()
        print('test data max :', stats['_max'].shape, stats['_max'])
        print('test data min :', stats['_min'].shape, stats['_min'])

    #     if save:
    #         file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    #         dirpath = os.path.dirname(graph_signal_matrix_filename)
    #         filename = os.path.join(dirpath,
    #                                 file + '_r' + str(self.num_of_hours) + '_d' + str(self.num_of_days) + '_w' + str(self.num_of_weeks))
    #         print('save file:', filename)
    #         np.savez_compressed(filename,
    #                             test_x=self.all_data['test']['x'], test_target=self.all_data['test']['target'],
    #                             test_timestamp=self.all_data['test']['timestamp'],
    #                             mean=self.all_data['stats']['_max'], std=self.all_data['stats']['_min']
    #                             )
        return self.all_data



    def predict_and_save_results(self, type):
        '''
        for transformerGCN
        :param self.net: nn.Module
        :param data_loader: torch.utils.data.utils.DataLoader
        :param data_target_tensor: tensor
        :param epoch: int
        :param self._max: (1, 1, 3, 1)
        :param self._min: (1, 1, 3, 1)
        :param self.params_path: the path for saving the results
        :return:
        '''
        self.net.train(False)  # ensure self.dropout layers are in test mode

        start_time = time()

        with torch.no_grad():

            self.data_target_tensor = self.data_target_tensor.transpose(-1, -2).cpu().numpy()

            loader_length = len(self.test_loader)  # nb of batch

            prediction = []

            input = []  # 存储所有batch的input

            start_time = time()

            for batch_index, batch_data in enumerate(self.test_loader):

                encoder_inputs, decoder_inputs, labels = batch_data

                encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

                decoder_inputs = decoder_inputs.transpose(-1, -2)  # (B, N, T, F)

                labels = labels.transpose(-1, -2)  # (B, N, T, F)

                predict_length = labels.shape[2]  # T

                # encode
                encoder_output = self.net.encode(encoder_inputs)
                input.append(encoder_inputs[:, :, :, 0:2].cpu().numpy())  # (batch, T', F)

                # decode
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                decoder_input_list = [decoder_start_inputs]

                # 按着时间步进行预测
                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    predict_output = self.net.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, predict_output]

                prediction.append(predict_output.detach().cpu().numpy())
                if batch_index % 100 == 0:
                    print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))
            self._max = np.transpose(self._max, (0,1,3,2)) #(B,N,T,F)
            self._min = np.transpose(self._min, (0,1,3,2)) #(B,N,T,F)
            print('test time on whole data:%.2fs' % (time() - start_time))
            input = np.concatenate(input, 0)
            input = self.re_max_min_normalization(input)

            prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
            prediction = self.re_max_min_normalization(prediction)
            self.data_target_tensor = self.re_max_min_normalization(self.data_target_tensor)

            print('input:', input.shape)
            print('prediction:', prediction.shape)
            print('data_target_tensor:', self.data_target_tensor.shape)

            # save outputs similar to libs.utils.predict_and_save_results
            try:
                output_filename = os.path.join(self.params_path, 'output_epoch_%s_%s.npz' % (self.best_epoch, type))
                np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=self.data_target_tensor)
            except Exception as e:
                print('warn: failed to save npz outputs:', e)

            # compute per-horizon and overall metrics (MAE/RMSE/MAPE)
            try:
                excel_list = []
                prediction_length = prediction.shape[2]
                for i in range(prediction_length):
                    assert self.data_target_tensor.shape[0] == prediction.shape[0]
                    mae_i = mean_absolute_error(self.data_target_tensor[:, :, i, 0], prediction[:, :, i, 0])
                    rmse_i = mean_squared_error(self.data_target_tensor[:, :, i, 0], prediction[:, :, i, 0]) ** 0.5
                    mape_i = masked_mape_np(self.data_target_tensor[:, :, i, 0], prediction[:, :, i, 0], 0)
                    print('Horizon %d -> MAE: %.4f | RMSE: %.4f | MAPE: %.4f' % (i+1, mae_i, rmse_i, mape_i))
                    excel_list.extend([mae_i, rmse_i, mape_i])

                mae = mean_absolute_error(self.data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
                rmse = mean_squared_error(self.data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
                mape = masked_mape_np(self.data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
                print('Overall -> MAE: %.4f | RMSE: %.4f | MAPE: %.4f' % (mae, rmse, mape))

                # cache last evaluation and optionally save
                self.last_metrics = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
                self.last_pred = prediction
                self.last_true = self.data_target_tensor

                try:
                    np.save(os.path.join(self.params_path, 'metrics.npy'), np.array([mae, rmse, mape]))
                    np.save(os.path.join(self.params_path, 'pred.npy'), prediction)
                    np.save(os.path.join(self.params_path, 'true.npy'), self.data_target_tensor)
                except Exception as e:
                    print('warn: failed to save metrics/pred/true:', e)
            except Exception as e:
                print('warn: metric computation failed:', e)

            return prediction


    def predict_main(self, type):
        '''
        在测试集上，测试指定epoch的效果
        :param epoch: int
        :param data_loader: torch.utils.data.utils.DataLoader
        :param data_target_tensor: tensor
        :param self._max: (1, 1, 3, 1)
        :param self._min: (1, 1, 3, 1)
        :param type: string
        :return:
        '''

        params_filename = os.path.join(self.params_path, 'epoch_%s.params' % self.best_epoch)
        print('load weight from:', params_filename, flush=True)

        self.net.load_state_dict(torch.load(params_filename))

        return self.predict_and_save_results(type)

    def test_main(self):
        """
        Mirror run.py's test-mode style: load best weights, run prediction,
        compute metrics, and persist outputs under params_path.

        Returns a dict with overall metrics (mae, rmse, mape).
        """
        _ = self.predict_main('test')
        return getattr(self, 'last_metrics', None)


if __name__ == "__main__":
    # read hyper-param settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/flatbushlarge5min.conf', type=str, help="configuration file path")
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config), flush=True)
    config.read(args.config)
    CT = Connector_1h(config, args.cuda)
    CT.predict_main('prediction')
