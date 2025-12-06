from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, EarlyStopping_withScaler, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import json
import warnings
import numpy as np
import logging
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.metrics import smape, mase, owa
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.checkpoint_loaded = False
        self.setting = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _get_default_checkpoint_path(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        return path


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.setting = setting
        checkpoint_path = self._get_default_checkpoint_path(setting)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping_withScaler(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, checkpoint_path, scaler=train_data.scaler)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = checkpoint_path
        self._load_checkpoint(best_model_path)
        return self.model
    
    def predict(self, setting, batch_x,):
        if self.setting is not None and self.setting != setting:
            print('warining: setting is not consistent with the one used in training or last prediction.')
        if not self.checkpoint_loaded:
            self.setting = setting
            checkpoint_path = self._get_default_checkpoint_path(setting)
            self._load_checkpoint(checkpoint_path)

        # reshape, scale and reshape back
        shape = batch_x.shape  # (batch, seq_len, features)
        flat = batch_x.reshape(shape[0] * shape[1], shape[2])
        flat = self.training_data_scaler.transform(flat)
        batch_x = flat.reshape(shape)

        # batch_x = self.training_data_scaler.transform(batch_x)
        batch_x = torch.from_numpy(batch_x)

        batch_x_mark = np.zeros((batch_x.shape[0], self.args.seq_len, 0)) # shape[0], seq_len, time_feature
        batch_x_mark = torch.from_numpy(batch_x_mark)
        self.model.eval()
        with torch.no_grad():
            batch_x = batch_x.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, None, None)
            else:
                outputs = self.model(batch_x, batch_x_mark, None, None)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, :]
            outputs = outputs.detach().cpu().numpy()
            # Inverse transform, return the data to the original scale
            shape = batch_x.shape
            if outputs.shape[-1] != batch_x.shape[-1]:
                outputs = np.tile(outputs, [1, 1, int(batch_x.shape[-1] / outputs.shape[-1])])
            outputs = self.training_data_scaler.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
            
            outputs = outputs[:, :, f_dim:]
            pred = outputs
        return pred
    
    def _load_checkpoint(self, path):
        model_path = path + '/' + 'checkpoint.pth'
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.checkpoint_loaded = True
        print(f"Model loaded from {path}")
        aux_state_path = path + '/' + 'aux_state.json'
        if os.path.exists(aux_state_path):
            with open(aux_state_path, 'r') as f:
                aux_state = json.load(f)
            scaler_state = aux_state.get('scaler', None)
            if scaler_state is not None:
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_state['mean'])
                scaler.scale_ = np.array(scaler_state['scale'])
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_samples_seen_ = 1
                self.training_data_scaler = scaler
                print("training_data_scaler state loaded from the checkpoint.")
            else:
                print("No scaler state found in the checkpoint.")
        else:
            print("No scaler state found in the checkpoint.")
        
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.setting = setting
            checkpoint_path = self._get_default_checkpoint_path(setting)
            self._load_checkpoint(checkpoint_path)
            
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # shape = batch_y.shape
                # print('batch_x', batch_x.shape, test_data.inverse_transform(batch_x.reshape(shape[0] * shape[1], -1)).reshape(shape), 'batch_y', batch_y.shape, batch_y)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # print('batch_x_mark', batch_x_mark.shape, batch_x_mark, 'batch_y_mark', batch_y_mark.shape, batch_y_mark)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
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
