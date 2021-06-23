import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT

from typing import List, Callable

from modules.distribution_output import StudentTOutput


def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class MLP_proba(nn.Module):
    def __init__(self,
                 num_hidden_dimensions: List[int],
                 hist_len: int,
                 pred_len: int,
                 distr_output: Callable = StudentTOutput(),
                 scaling: Callable=mean_abs_scaling):
        super(MLP_proba, self).__init__()

        self.num_hidden_dimensions = num_hidden_dimensions
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.distr_output = distr_output
        self.scaling = scaling

        modules = []
        dims = self.num_hidden_dimensions

        for i, units in enumerate(dims):
            if i == 0:
                input_size = hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]

        # modules.append(nn.Linear(dims[-1], pred_len))
        modules.append(nn.Linear(dims[-1], pred_len * dims[-1]))

        self.mlp = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(dims[-1])


    def forward(self, x):
        scale = self.scaling(x)
        x = x / scale
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.reshape(x.shape[0], self.pred_len, x.shape[1], -1)
        distr_args = self.args_proj(x)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        return distr
