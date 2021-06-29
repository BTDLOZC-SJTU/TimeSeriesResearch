import torch
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU
from torch import Tensor

from typing import List, Callable, Optional

from utils.embed import DataEmbedding
from modules.distribution_output import StudentTOutput


def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class DeepAR(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 d_model: int,
                 hist_len: int,
                 pred_len: int,
                 num_layers: int = 1,
                 hidden_size: int = 40,
                 embedding_dim: int = 10,
                 cell_type: str = 'GRU',
                 dropout_rate: float = 0.0,
                 freq: str = 'H',
                 num_parallel_samples: int = 100,
                 distr_output: Callable = StudentTOutput(),
                 scaling: Callable = mean_abs_scaling
                 ):
        super(DeepAR, self).__init__()

        assert (
                c_in==c_out
        ), "Auto-regressive model should have same input dimension and output dimension"

        freq_map = {'Y': 1, 'M': 1, 'D': 3, 'H': 4, 'T': 5, 'S': 6}

        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim * freq_map[freq.upper()]
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.freq = freq

        self.num_parallel_samples = num_parallel_samples
        self.distr_output = distr_output
        self.scaling = scaling

        self.time_feat_embedding = nn.Linear(freq_map[freq.upper()], self.embedding_dim)

        rnn_cell_map = {'RNN': RNN, 'LSTM': LSTM, 'GRU': GRU}

        # RNN input_size =
        # c_in(target feature dimension) +
        # self.embedding_dim(time feature dimension)
        self.rnn = rnn_cell_map[cell_type](input_size=c_in + self.embedding_dim,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           dropout=dropout_rate,
                                           batch_first=True)

        self.args_proj = self.distr_output.get_args_proj(hidden_size)

    def unroll_encoder(self,
                       x: Tensor,  # (batch_size, hist_len, c_in)
                       x_mark: Tensor,  # (batch_size, hist_len, time_feat_dim)
                       y: Optional[Tensor],  # (batch_size, pred_len, c_out)
                       y_mark: Optional[Tensor],  # (batch_size, pred_len, time_feat_dim)
                       ):
        if y is None or y_mark is None:
            time_feat = x_mark
            sequence = x
        else:
            time_feat = torch.cat((x_mark, y_mark), dim=1)
            sequence = torch.cat((x, y), dim=1)

        time_feat = self.time_feat_embedding(time_feat)
        scale = self.scaling(sequence)
        sequence = sequence / scale

        inputs = torch.cat((sequence, time_feat), dim=-1)

        # inputs: (input, h_0) h_0 (num_layers, batch_size, hidden_size) zero without initialize
        # outputs (batch_size, hist_len, hidden_size) all state in sequence
        # state (num_layers, batch_size, hidden_size) final hidden state
        outputs, state = self.rnn(inputs)

        return outputs, state, scale

    def sampling_decoder(self,
                         x: Tensor,
                         x_mark: Tensor,
                         y_mark: Tensor,
                         begin_state: Tensor,
                         scale: Tensor,
                         ):

        # increase parallelism
        # blows-up the dimension of each tensor to
        # (batch_size * self.num_parallel_samples) for increasing parallelism

        repeated_x = x.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_y_mark = y_mark.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_y_mark = self.time_feat_embedding(repeated_y_mark)
        repeated_scale = scale.repeat_interleave(repeats=self.num_parallel_samples, dim=0)

        # state (num_layers, batch_size, hidden_size) batch_size dim lies in dim 1
        if self.cell_type == 'LSTM':
            repeated_states = [
                s.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
                for s in begin_state
            ]
        else:
            repeated_states = begin_state.repeat_interleave(repeats=self.num_parallel_samples, dim=1)

        future_samples = []

        # (batch_size * num_samples, 1, c_in)
        tgt = repeated_x[:, -1: , :]
        for k in range(self.pred_len):
            # (batch_size * num_samples, 1, c_in + time_feat_dim)
            decoder_inputs = torch.cat((tgt, repeated_y_mark[:, k, :].unsqueeze(1)), dim=-1)

            rnn_outputs, repeated_states = self.rnn(decoder_inputs, repeated_states)

            distr_args = self.args_proj(rnn_outputs.unsqueeze(2))
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

            # (batch_size * num_samples, 1, c_out)
            new_samples = distr.sample()
            future_samples.append(new_samples)
            tgt = new_samples

        # (batch_size * num_samples, pred_len, c_out)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, pred_len, c_out)
        return samples.reshape(
                (self.num_parallel_samples, self.pred_len, self.c_out)
        )

    def forward(self,
                x: Tensor,
                x_mark: Tensor,
                y: Optional[Tensor],
                y_mark: Optional[Tensor],
                mode: bool):
        if mode:
            # train mode
            rnn_outputs, _, scale = self.unroll_encoder(x, x_mark, y, y_mark)
            distr_args = self.args_proj(rnn_outputs.unsqueeze(2))

            return self.distr_output.distribution(distr_args, scale=scale)

        else:
            # eval mode
            # in this mode, y = None

            # get begin state of decoder
            _, state, scale = self.unroll_encoder(x, x_mark, y, y_mark)

            return self.sampling_decoder(x, x_mark, y_mark, state, scale)
