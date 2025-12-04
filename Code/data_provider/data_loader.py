import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from scipy.interpolate import interp1d
from torch.utils.data import ConcatDataset
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        df_raw = df_raw[['date'] + cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print(f"batch_x: {seq_x.shape}, batch_y: {seq_y.shape}")
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Traffic_Multivariate(Dataset):
    def __init__(self, args, root_path='traffic/edge_merge/', flag='train', size=None, data_path='id_1-2.csv',
                 target='speed', scale=True, seasonal_patterns=None, features='M', timeenc=0, freq='h'):
        # 序列长度设置 [历史序列长度, 标签序列长度, 预测序列长度]
        if size is None:
            self.seq_len = 24  # 例如：用24个时间间隔的历史数据
            self.label_len = 12  # 标签序列长度（用于辅助预测）
            self.pred_len = 6   # 预测未来6个时间间隔
        else:
            self.seq_len, self.label_len, self.pred_len = size
        
        # 验证数据集类型
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        
        self.args = args
        self.target = target  
        self.scale = scale   
        self.root_path = root_path
        self.data_path = data_path
        # 读取并处理数据
        self.__read_data__()

    def __read_data__(self):
        # 1. 读取数据（假设root_path是单个ID的CSV文件路径）
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # 2. 筛选特征：移除不需要的列
        # 需要移除的列
        drop_cols = [
            'file_id', 'file_name', 'interval_begin', 
            'interval_end', 'interval_id', 'id'
        ]
        # 确保待移除的列存在于数据中
        drop_cols = [col for col in drop_cols if col in df_raw.columns]
        df_filtered = df_raw.drop(columns=drop_cols, errors='ignore')
        
        # 3. 处理缺失值：用插值法填充（线性插值）
        # 先将非数值列排除（仅对数值列插值）
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 获取非缺失值的索引和值
            non_na_idx = df_filtered[col].dropna().index
            non_na_vals = df_filtered.loc[non_na_idx, col].values
            
            if len(non_na_vals) < 2:
                # 若有效数据太少，用均值填充
                df_filtered[col] = df_filtered[col].fillna(df_filtered[col].mean())
            else:
                # 线性插值
                f = interp1d(non_na_idx, non_na_vals, kind='linear', bounds_error=False, fill_value="extrapolate")
                df_filtered[col] = f(df_filtered.index)
        
        # 4. 划分训练/验证/测试集
        total_len = len(df_filtered)
        num_train = int(total_len * 0.7)
        num_test = int(total_len * 0.2)
        num_val = total_len - num_train - num_test
        
        # 数据集边界（确保序列长度足够）
        border1s = [
            0,                                  # 训练集起始
            num_train - self.seq_len,           # 验证集起始
            total_len - num_test - self.seq_len # 测试集起始
        ]
        border2s = [
            num_train,                          # 训练集结束
            num_train + num_val,                # 验证集结束
            total_len                           # 测试集结束
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 5. 提取特征和目标变量
        # 所有数值列作为输入特征（包含target列，后续会自动对齐）
        self.features = df_filtered.columns.tolist()
        df_data = df_filtered[self.features]
        
        # 6. 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            # 用训练集数据拟合标准化器
            train_data = df_data.iloc[:num_train].values
            self.scaler.fit(train_data)
            # 对所有数据标准化
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 7. 划分输入和输出（输入输出用相同的数据，通过索引截取实现序列预测）
        self.data_x = data[border1:border2]  # 输入数据（历史序列）
        self.data_y = data[border1:border2]  # 输出数据（包含标签和预测序列）

    def __getitem__(self, index):
        # 计算序列索引
        s_begin = index  # 输入序列起始
        s_end = s_begin + self.seq_len  # 输入序列结束
        r_begin = s_end - self.label_len  # 标签序列起始（用于辅助预测）
        r_end = r_begin + self.label_len + self.pred_len  # 输出序列结束（标签+预测）
        
        # 截取序列
        seq_x = self.data_x[s_begin:s_end]  # 历史输入序列 (seq_len, features)
        seq_y = self.data_y[r_begin:r_end]  # 输出序列 (label_len + pred_len, features)
        
        # 由于移除了date编码，这里返回空的时间标记（或根据需要返回None）
        seq_x_mark = np.zeros((self.seq_len, 0))  # 空时间标记
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 0))  # 空时间标记
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 计算可生成的样本数
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # 反标准化，用于将预测结果恢复原始尺度
        return self.scaler.inverse_transform(data)


class Dataset_Traffic_Singlevariate(Dataset):
    def __init__(self, args, root_path='traffic/edge_merge/', flag='train', size=None, data_path='id_1-2.csv',
                 target='speed', scale=True, seasonal_patterns=None, features='M', timeenc=0, freq='h'):
        # 序列长度设置 [历史序列长度, 标签序列长度, 预测序列长度]
        if size is None:
            self.seq_len = 24  # 例如：用24个时间间隔的历史数据
            self.label_len = 12  # 标签序列长度（用于辅助预测）
            self.pred_len = 6   # 预测未来6个时间间隔
        else:
            self.seq_len, self.label_len, self.pred_len = size
        
        # 验证数据集类型
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        
        self.args = args
        self.target = target  # 预测目标：speed
        self.scale = scale    # 是否标准化
        self.root_path = root_path
        self.data_path = data_path
        # 读取并处理数据
        self.__read_data__()

    def __read_data__(self):
        # 1. 读取数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 2. 筛选特征：移除不需要的列
        drop_cols = [
            'file_id', 'file_name', 'interval_begin', 
            'interval_end', 'interval_id', 'id'
        ]
        drop_cols = [col for col in drop_cols if col in df_raw.columns]
        df_filtered = df_raw.drop(columns=drop_cols, errors='ignore')
        
        # 3. 处理缺失值：用插值法填充（线性插值）
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            non_na_idx = df_filtered[col].dropna().index
            non_na_vals = df_filtered.loc[non_na_idx, col].values
            
            if len(non_na_vals) < 2:
                df_filtered[col] = df_filtered[col].fillna(df_filtered[col].mean())
            else:
                f = interp1d(non_na_idx, non_na_vals, kind='linear', bounds_error=False, fill_value="extrapolate")
                df_filtered[col] = f(df_filtered.index)
        
        # 4. 划分训练/验证/测试集
        total_len = len(df_filtered)
        num_train = int(total_len * 0.7)
        num_test = int(total_len * 0.2)
        num_val = total_len - num_train - num_test
        
        # 数据集边界（确保序列长度足够）
        border1s = [
            0,                                  # 训练集起始
            num_train - self.seq_len,           # 验证集起始
            total_len - num_test - self.seq_len # 测试集起始
        ]
        border2s = [
            num_train,                          # 训练集结束
            num_train + num_val,                # 验证集结束
            total_len                           # 测试集结束
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 5. 提取特征和目标变量
        self.features = df_filtered.columns.tolist()
        # 确保目标变量在特征中
        assert self.target in self.features, f"目标变量 {self.target} 不在数据列中"
        
        # 输入特征：所有数值列
        df_features = df_filtered[self.features]
        # 目标变量：仅提取 target 列（用于输出）
        df_target = df_filtered[[self.target]]  # 保持二维结构 (len, 1)
        
        # 6. 数据标准化
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # 单独用于目标变量的标准化器
        
        if self.scale:
            # 输入特征标准化（用训练集拟合）
            train_feat_data = df_features.iloc[:num_train].values
            self.scaler.fit(train_feat_data)
            feat_data = self.scaler.transform(df_features.values)
            
            # 目标变量单独标准化（用训练集拟合）
            train_target_data = df_target.iloc[:num_train].values
            self.target_scaler.fit(train_target_data)
            target_data = self.target_scaler.transform(df_target.values)
        else:
            feat_data = df_features.values
            target_data = df_target.values
        
        # 7. 划分输入和输出（输入用全部特征，输出仅用目标变量）
        self.data_x = feat_data[border1:border2]  # 输入：所有特征 (len, n_features)
        self.data_y = target_data[border1:border2]  # 输出：仅目标变量 (len, 1)

    def __getitem__(self, index):
        # 计算序列索引
        s_begin = index  # 输入序列起始
        s_end = s_begin + self.seq_len  # 输入序列结束
        r_begin = s_end - self.label_len  # 标签序列起始
        r_end = r_begin + self.label_len + self.pred_len  # 输出序列结束（标签+预测）
        
        # 截取序列
        seq_x = self.data_x[s_begin:s_end]  # 历史输入序列 (seq_len, n_features)
        seq_y = self.data_y[r_begin:r_end]  # 输出序列（仅目标变量）(label_len + pred_len, 1)
        
        # 空时间标记
        seq_x_mark = np.zeros((self.seq_len, 0))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 0))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 计算可生成的样本数
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # 反标准化目标变量（恢复原始尺度）
        return self.target_scaler.inverse_transform(data)

class Dataset_Traffic_Merge(Dataset):
    def __init__(self, args, root_path, data_path, features,seasonal_patterns, freq, timeenc, flag='train', size=None, target='speed', scale=True):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.size = size
        self.target = target
        self.scale = scale
        
        # 存储所有单个数据集的实例（对应flag的子集）
        self.datasets = []
        self.dataset_paths = []  # 存储每个数据集的路径，方便后续反标准化
        
        # 遍历文件夹中的所有CSV文件
        for filename in os.listdir(root_path):
            if filename.endswith('.csv'):
                # data_path = os.path.join(root_path, filename)
                # 为每个CSV文件创建对应flag的数据集实例
                dataset = Dataset_Traffic_Singlevariate(
                    args=args,
                    data_path=filename,
                    root_path=self.root_path,
                    flag=self.flag,
                    size=self.size,
                    target=self.target,
                    scale=self.scale
                )
                if len(dataset) > 0:  # 只添加非空数据集
                    self.datasets.append(dataset)
                    self.dataset_paths.append(filename)
        
        # 合并所有数据集（训练/验证/测试都合并）
        self.union_dataset = ConcatDataset(self.datasets)
        
        # 预计算每个样本属于哪个原始数据集（用于反标准化）
        self.sample_to_dataset_idx = []
        for idx, dataset in enumerate(self.datasets):
            self.sample_to_dataset_idx.extend([idx] * len(dataset))

    def __getitem__(self, index):
        # 从合并的数据集获取样本
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.union_dataset[index]
        # 返回样本 + 该样本所属的原始数据集索引（用于反标准化）
        return seq_x, seq_y, seq_x_mark, seq_y_mark
        # return seq_x, seq_y, seq_x_mark, seq_y_mark, self.sample_to_dataset_idx[index]

    def __len__(self):
        return len(self.union_dataset)

    def inverse_transform(self, data, dataset_idx):
        # 根据数据集索引，使用对应数据集的scaler进行反标准化
        return self.datasets[dataset_idx].inverse_transform(data)