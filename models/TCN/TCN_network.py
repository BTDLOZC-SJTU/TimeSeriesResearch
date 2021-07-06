import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from utils.embed import DataEmbedding

from typing import List, Tuple


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dropout_rate: float):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.chomp1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.chomp2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        if self.downsample is not None:
            inp = self.downsample(inp)
        return self.relu(x + inp)


class TCN(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 num_hidden_dimensions: List[int],
                 hist_len: int,
                 pred_len: int,
                 dilation_base: int = 2,
                 kernel_size: int = 2,
                 dropout_rate: float = 0.2):
        super(TCN, self).__init__()

        self.hist_len = hist_len
        self.pred_len = pred_len
        assert hist_len >= pred_len,\
            "history length should larger or equal to prediction length"

        layers = []
        dims = num_hidden_dimensions
        num_levels = len(dims)
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = c_in if i == 0 else dims[i - 1]
            out_channels = dims[i]
            layers += [
                TemporalBlock(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout_rate=dropout_rate)
            ]
        self.temporal_conv_net = nn.Sequential(*layers)
        self.projection = nn.Linear(dims[-1], c_out)

    def forward(self, x, x_mark, y_mark):
        x = self.temporal_conv_net(x.transpose(1, 2)).transpose(1, 2)
        x = self.projection(x)
        return x[:, -self.pred_len: ,:]














































