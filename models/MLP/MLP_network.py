import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class MLP(nn.Module):
    def __init__(self,
                 num_hidden_dimensions: List[int],
                 hist_len: int,
                 pred_len: int,):
        super(MLP, self).__init__()

        self.num_hidden_dimensions = num_hidden_dimensions
        self.hist_len = hist_len
        self.pred_len = pred_len

        modules = []
        dims = self.num_hidden_dimensions

        for i, units in enumerate(dims):
            if i == 0:
                input_size = hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]

        modules.append(nn.Linear(dims[-1], pred_len))

        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        return x
