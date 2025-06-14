import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='MK2000_with_macro_and_volatility.csv',
                 target='Volatility', scale=True, timeenc=0, freq='h'):
        """
        Custom Dataset for time series data
        """
        if size is None:
            self.seq_len = 336
            self.label_len = 0
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val'], "flag must be 'train', 'val', or 'test'"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # Load and process data
        self.__read_data__()

    def __read_data__(self):
        """
        Read and preprocess data
        """
        self.scaler = StandardScaler()
        try:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.data_path}. Ensure the path is correct.") from e

        if self.target not in df_raw.columns:
            raise ValueError(f"Target column '{self.target}' not found in the dataset. Available columns: {list(df_raw.columns)}")

        if 'Date' in df_raw.columns:
            df_raw.rename(columns={'Date': 'date'}, inplace=True)
        if 'date' not in df_raw.columns:
            raise ValueError("Dataset must contain a 'date' column for timestamp information.")

        df_raw['date'] = pd.to_datetime(df_raw['date'])

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Invalid features setting: {self.features}. Must be 'S', 'M', or 'MS'.")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
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

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    #def __len__(self):
        #return len(self.data_x) - self.seq_len - self.pred_len + 1
    def __len__(self):
        total_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if total_length <= 0:
            suggested_pred_len = len(self.data_x) - self.seq_len - 1
            raise ValueError(
                f"Dataset length is insufficient: len(data_x)={len(self.data_x)}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}. "
                f"Try reducing pred_len to a maximum of {suggested_pred_len}."
            )
        return total_length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset_Custom):
    pass


class Dataset_ETT_minute(Dataset_Custom):
    pass


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='MK2000_with_macro_and_volatility.csv',
                 target='Volatility', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        """
        Dataset for prediction
        """
        if size is None:
            self.seq_len = 336
            self.label_len = 0
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if 'Date' in df_raw.columns:
            df_raw.rename(columns={'Date': 'date'}, inplace=True)
        if 'date' not in df_raw.columns:
            raise ValueError("Dataset must contain a 'date' column for timestamp information.")

        df_raw['date'] = pd.to_datetime(df_raw['date'])

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp['date'] = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_raw[[self.target]].values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    '''def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Return total samples
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1'''

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if s_end > len(self.data_x) or r_end > len(self.data_x):
            raise IndexError(
                f"Index out of range: s_end={s_end}, r_end={r_end}, len(data_x)={len(self.data_x)}"
            )

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        total_samples = len(self.data_x) - self.seq_len - self.pred_len + 1
        if total_samples < 1:
            raise ValueError(
                f"Dataset length is insufficient: len(data_x)={len(self.data_x)}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}. Adjust seq_len or pred_len."
            )
        return total_samples




class Dataset_ETT_minute(Dataset_ETT_hour):
    """
    Dataset for minute-level ETT data
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq)

