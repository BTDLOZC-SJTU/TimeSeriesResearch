import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Callable

from utils.embed import DataEmbedding
from modules.distribution_output import StudentTOutput


def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class MLP_proba(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 d_model: int,
                 num_hidden_dimensions: List[int],
                 hist_len: int,
                 pred_len: int,
                 freq: str = 'H',
                 use_time_feat: bool = True,
                 distr_output: Callable = StudentTOutput(),
                 scaling: Callable=mean_abs_scaling):
        super(MLP_proba, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.num_hidden_dimensions = num_hidden_dimensions
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.freq = freq
        self.distr_output = distr_output
        self.scaling = scaling

        # Embedding
        self.embedding = DataEmbedding(c_in, d_model, freq, use_time_feat=use_time_feat)

        modules = []
        dims = self.num_hidden_dimensions

        for i, units in enumerate(dims):
            if i == 0:
                input_size = hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]

        modules.append(nn.Linear(dims[-1], pred_len * dims[-1]))

        self.projection = nn.Linear(d_model, c_out)

        self.mlp = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(dims[-1])


    def forward(self, x, x_mark, y_mark):
        x = self.embedding(x, x_mark)
        scale = self.scaling(x)
        x = x / scale
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], self.pred_len, x.shape[1], -1)
        distr_args = self.args_proj(x)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        return distr
