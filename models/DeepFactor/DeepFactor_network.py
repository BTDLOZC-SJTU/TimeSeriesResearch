import torch
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU
from torch import Tensor
from pandas.tseries.frequencies import to_offset
from typing import List, Callable

from modules.distribution_output import StudentTOutput


class DeepFactor(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 d_model: int,
                 hist_len: int,
                 pred_len: int,
                 num_hidden_global: int = 50,
                 num_layers_global: int = 1,
                 num_factors: int = 10,
                 num_hidden_local: int = 5,
                 num_layers_local: int = 1,
                 embedding_dim: int = 10,
                 cell_type: str = 'GRU',
                 freq: str = 'H',
                 use_time_feat: bool = True,
                 ):
        super(DeepFactor, self).__init__()

        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}

        self.c_in = c_in
        self.c_out = c_out
        self.freq = to_offset(freq)
        self.embedding_dim = embedding_dim * freq_map[self.freq.name]

        self.global_model = RecurrentModule(cell_type=cell_type.upper(),
                                            input_size=freq_map[self.freq.name],
                                            hidden_size=num_hidden_global,
                                            num_layers=num_layers_global,
                                            num_factors=num_factors,
                                            bidirectional=True)

        self.local_model = RecurrentModule(cell_type=cell_type.upper(),
                                           input_size=freq_map[self.freq.name] + self.embedding_dim,
                                           hidden_size=num_hidden_global,
                                           num_layers=num_layers_local,
                                           num_factors=1,
                                           bidirectional=True)


        self.assemble_features_embedding = nn.Linear(1, self.embedding_dim)
        self.loading = nn.Linear(self.embedding_dim, num_factors, bias=False)

        self.freq = freq

    def assemble_features(self, x_mark: Tensor):
        latent_feat = torch.zeros(size=(x_mark.shape[0], 1)).to(x_mark.device)  # (batch_size, 1)
        embed_feat = self.assemble_features_embedding(latent_feat)  # (batch_size, num_features * embedding_size)

        helper_ones = torch.ones(size=(x_mark.shape[0], x_mark.shape[1], 1)).to(x_mark.device)
        repeated_cat = torch.bmm(helper_ones, embed_feat.unsqueeze(1))
        local_input = torch.cat((repeated_cat, x_mark), dim=2)

        return embed_feat, local_input

    def forward(self, x, x_mark, y_mark):
        embed_feat, local_input = self.assemble_features(x_mark)

        loadings = self.loading(embed_feat)
        global_factors = self.global_model(x_mark)

        fixed_effect = torch.bmm(global_factors, loadings.unsqueeze(2)) # (batch_size, history_length, 1)
        fixed_effect = torch.exp(fixed_effect)

        random_effect = torch.log(
            torch.exp(self.local_model(local_input)) + 1.0
        )

        return fixed_effect, random_effect


class RecurrentModule(nn.Module):
    def __init__(self,
                 cell_type: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_factors: int,
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
        if not bidirectional:
            self.projection = nn.Linear(hidden_size, num_factors)
        else:
            self.projection = nn.Linear(2 * hidden_size, num_factors)

    def forward(self, x):
        x, _  = self.rnn(x)
        x = self.projection(x)

        return x