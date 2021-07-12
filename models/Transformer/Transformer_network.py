import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from pandas.tseries.frequencies import to_offset

from utils.embed import DataEmbedding

from typing import List, Tuple


class Transformer(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 hist_len: int,
                 cntx_len: int,
                 pred_int: int,
                 d_model: int = 512,
                 n_head: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 embedding_dim: int = 10,
                 freq: str = 'H'):
        super(Transformer, self).__init__()

        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}

        self.c_in = c_in
        self.c_out = c_out
        self.hist_len = hist_len
        self.cntx_len = cntx_len
        self.pred_len = pred_int
        self.freq = to_offset(freq)
        self.embedding_dim = embedding_dim

        self.time_feat_embedding = nn.Linear(freq_map[self.freq.name], self.embedding_dim)
        self.enc_embedding = nn.Linear(self.embedding_dim + c_in, d_model)
        self.dec_embedding = nn.Linear(self.embedding_dim + c_in, d_model)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout_rate,
                                          activation=activation)

        self.projection = nn.Linear(d_model, c_out)

    def create_inputs(self, x, x_mark, y_mark):

        mark_inp = torch.cat((x_mark, y_mark), dim=1)
        mark_inp = self.time_feat_embedding(mark_inp)

        enc_inp = torch.cat((x, mark_inp[:, : self.hist_len, :]), dim=-1)
        dec_inp = torch.cat((x[:, -self.cntx_len: ,:],
                             torch.zeros(size=(x.shape[0], self.pred_len, x.shape[2])).to(x.device)),
                      dim=1)  # (batch_size, cntx_len + pred_len, c_in)
        dec_inp = torch.cat((dec_inp, mark_inp[:, -self.cntx_len - self.pred_len: , :]), dim=-1)

        return enc_inp, dec_inp

    def forward(self, x, x_mark, y_mark):
        enc_inp, dec_inp = self.create_inputs(x, x_mark, y_mark)
        enc_inp = self.enc_embedding(enc_inp)
        dec_inp = self.dec_embedding(dec_inp)

        out = self.transformer(enc_inp.transpose(0, 1), dec_inp.transpose(0, 1))

        out = self.projection(out).transpose(0, 1)
        return out[:, -self.pred_len :, :]
