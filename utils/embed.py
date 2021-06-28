import torch
import torch.nn as nn


class TokenLinearEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenLinearEmbedding, self).__init__()

        self.tokenLinear = nn.Linear(c_in, d_model)

    def forward(self, x):
        x = self.tokenLinear(x)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='H'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'Y': 1, 'M': 1, 'D': 3, 'H': 4, 'T': 5, 'S': 6}
        d_inp = freq_map[freq.upper()]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, freq='H', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenLinearEmbedding(c_in, d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)

        return x