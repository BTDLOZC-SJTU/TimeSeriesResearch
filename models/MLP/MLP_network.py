import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class MLP(nn.Module):
    def __init__(self,
                 num_hidden_dimension: List[int],
                 hist_len: int,
                 pred_len: int):
        super(MLP, self).__init__()

        self.num_hidden_dimension = num_hidden_dimension
        self.hist_len = hist_len
        self.pred_len = pred_len

        modules = []
        dims = self.num_hidden_dimension

        for i, units in enumerate(dims[: - 1]):
            if i == 0:
                input_size = self.hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]

        modules.append(nn.Linear(dims[-1], pred_len))

        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        x = self.mlp(x)
        return x
