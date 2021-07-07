import torch
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU


from pandas.tseries.frequencies import to_offset


class RecurrentModule(nn.Module):
    def __init__(self,
                 cell_type: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super(RecurrentModule, self).__init__()

        rnn_cell_map = {'RNN': RNN, 'LSTM': LSTM, 'GRU': GRU}
        self.rnn = rnn_cell_map[cell_type](input_size=input_size,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           bidirectional=bidirectional,
                                           batch_first=True)

    def forward(self, x):
        return self.rnn(x)


class LSTNet(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 hist_len: int,
                 pred_len: int,
                 out_channels: int,
                 kernel_size: int,
                 cell_type: str,
                 hidden_size: int,
                 num_layers:int,
                 skip_cell_type: str,
                 skip_hidden_size: int,
                 skip_num_layers: int,
                 skip_size: int,
                 ar_window: int,
                 embedding_dim: int = 10,
                 dropout_rate: float = 0.2,
                 freq: str = 'H'):
        """
        :param c_in:
        :param c_out:
        :param hist_len:
        :param pred_len:
        :param out_channels: Number of channels for first layer Conv2D
        :param kernel_size:
        :param cell_type:
        :param hidden_size:
        :param num_layers:
        :param skip_cell_type:
        :param skip_hidden_size:
        :param skip_num_layers:
        :param skip_size: Skip size for skip-RNN layers
        :param ar_window: Auto-regressive window size for the linear part
        :param embedding_dim:
        :param freq:
        """
        super(LSTNet, self).__init__()

        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}

        self.c_in = c_in
        self.c_out = c_out
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.skip_cell_type = skip_cell_type
        self.skip_hidden_size = skip_hidden_size
        self.skip_num_layers = skip_num_layers
        self.skip_size = skip_size
        self.skip_num = int((hist_len - kernel_size) / skip_size)

        self.ar_window = ar_window

        self.embedding_dim = embedding_dim
        self.freq = to_offset(freq)
        self.dropout_rate = dropout_rate

        # Embedding Module
        self.time_feat_embedding = nn.Linear(freq_map[self.freq.name], self.embedding_dim)

        # Convolution Module
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, c_in + embedding_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # RNN Module
        self.rnn = RecurrentModule(cell_type=cell_type,
                                   input_size=out_channels,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout_rate)

        # Skip-RNN Module
        self.skip_rnn = RecurrentModule(cell_type=skip_cell_type,
                                        input_size=out_channels,
                                        hidden_size=skip_hidden_size,
                                        num_layers=skip_num_layers,
                                        dropout=dropout_rate)

        self.projection = nn.Linear(hidden_size + skip_hidden_size * skip_size,
                                    pred_len * c_out)

        # Highway Module
        self.highway = nn.Linear(ar_window * (c_in + embedding_dim), pred_len * c_out)



    def forward(self, x, x_mark, y_mark):
        # Embedding Process
        x_mark_emb = self.time_feat_embedding(x_mark)
        x = torch.cat((x, x_mark_emb), dim=-1) # (batch_size, hist_len, c_in + embedding_dim)

        # Convolution Process
        c = x.unsqueeze(1) # (batch_size, 1, hist_len, c_in + embedding_dim)
        c = self.conv(c) # (batch_size, out_channels, hist_len - kernel_size, 1)
        c = c.squeeze(3) # (batch_size, out_channels, hist_len - kernel_size)
        c = self.relu(c)
        c = self.dropout(c)

        # RNN Process
        r = c.transpose(1, 2) # (batch_size, hist_len - kernel_size, out_channels)
        _, r = self.rnn(r) # (D * num_layers, batch_size, hidden_size)
        r = r.squeeze(0) # (batch_size, hidden_size)

        # Skip-RNN Process
        s = c[:, :, int(-self.skip_num * self.skip_size): ].contiguous()
        # (batch_size, out_channels, skip_num, skip_size)
        s = s.view(x.shape[0], self.out_channels, self.skip_num, self.skip_size)
        s = s.permute(0, 3, 2, 1).contiguous()
        s = s.view(x.shape[0] * self.skip_size, self.skip_num, self.out_channels)
        _, s = self.skip_rnn(s) # (D * num_layers, batch_size * skip_size, skip_hidden_size)
        s = s.view(x.shape[0], self.skip_size * self.skip_hidden_size)

        r = torch.cat((r, s), dim=1)
        r = self.projection(r) # (batch_size, pred_len * c_out)
        r = r.view(x.shape[0], self.pred_len, self.c_out)

        # Highway Process
        z = x[:, -self.ar_window: , : ]
        z = z.view(x.shape[0], self.ar_window * x.shape[2])
        z = self.highway(z) # (batch_size, pred_len * c_out)
        z = z.view(x.shape[0], self.pred_len, self.c_out)

        ret = r + z
        return ret




















