import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import timeFeatures

import warnings
warnings.filterwarnings('ignore')


class Dataset_TS(Dataset):
    def __init__(self,
                 data_path: str,
                 freq: str,
                 start_date: str,
                 flag: str,
                 hist_len: int,
                 pred_len: int,
                 cols: Optional[List[int]] = None,
                 train_valid_test_weights: Tuple = (0.7, 0.1, 0.2)
                 ):
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 2}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.start_date = start_date
        self.freq = freq
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.cols = cols
        self.train_valid_test_weights = train_valid_test_weights
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path, usecols=self.cols, header=None)
        num_time_steps = df_raw.shape[0]
        df_stamp = pd.date_range(start=self.start_date, periods=num_time_steps, freq=self.freq)

        train_len = int(num_time_steps * 0.7)
        test_len = int(num_time_steps * 0.2)
        valid_len = num_time_steps - train_len - test_len

        border1s = [0, train_len, train_len + valid_len]
        border2s = [train_len, train_len + valid_len, num_time_steps]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw.values
        data_stamp = timeFeatures(df_stamp, freq=self.freq)

        self.data_x = data[border1: border2]
        self.data_y = data[border1: border2]
        self.data_stamp = data_stamp[border1: border2]

    def __getitem__(self, index: int):
        h_begin = index
        h_end = h_begin + self.hist_len
        p_begin = h_end
        p_end = p_begin + self.pred_len

        seq_x = self.data_x[h_begin: h_end]
        seq_y = self.data_y[p_begin: p_end]
        seq_x_mark = self.data_stamp[h_begin: h_end]
        seq_y_mark = self.data_stamp[p_begin: p_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.hist_len - self.pred_len + 1





