import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import numpy as np

from typing import Optional



class ScaledDotProductAttention(nn.Module):
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None):
        emb_dim = query.shape[-1]
        query = query.permute(0, 2, 1, 3) # (batch_size, num_heads, tgt_len, dim)
        key = key.permute(0, 2, 1, 3)  # (batch_size, num_heads, src_len, dim)
        value = value.permute(0, 2, 1, 3)  # (batch_size, num_heads, src_len, dim)

        scores = query.matmul(key.transpose(-1, -2)) / np.sqrt(emb_dim) # (batch_size, num_heads, tgt_len, src_len)
        if attn_mask:
            scores = scores.masked_fill(attn_mask, -np.inf)

        attn_output_weights = F.dropout(F.softmax(scores, dim=-1))
        attn_output = attn_output_weights.matmul(value) # (batch_size, num_heads, tgt_len, dim)

        return attn_output, attn_output_weights


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.,
                 bias: bool = True,
                 kv_bias: bool = False,
                 k_dim: int = None,
                 v_dim: int = None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = embed_dim // num_heads

        self.bias = bias
        self.kv_bias = kv_bias

        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.q_proj_layer = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_layer = nn.Linear(embed_dim, self.k_dim, bias=kv_bias)
        self.v_proj_layer = nn.Linear(embed_dim, self.v_dim, bias=kv_bias)

        self.attention_layer = ScaledDotProductAttention()

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj_layer.weight.data)
        xavier_uniform_(self.k_proj_layer.weight.data)
        xavier_uniform_(self.v_proj_layer.weight.data)

        if self.bias:
            constant_(self.q_proj_layer.bias.data, 0.)

        if self.kv_bias:
            xavier_normal_(self.k_proj_layer.bias.data)
            xavier_normal_(self.v_proj_layer.bias.data)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None):
        """

        :param query: (batch_size, tgt_len, emb_dim)
        :param key: (batch_size, src_len, emb_dim)
        :param value: (batch_size, src_len, emb_dim)
        :param attn_mask: (tgt_len, src_len)

        :return:
        :output attn_output: (batch_size, num_heads, tgt_len, emb_dim)
        :output attn_output_weights: (batch_size, num_heads, tgt_len, src_len)
        """
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        num_heads = self.num_heads

        q = self.q_proj_layer(query).view(batch_size, tgt_len, num_heads, -1)
        k = self.k_proj_layer(key).view(batch_size, src_len, num_heads, -1)
        v = self.v_proj_layer(value).view(batch_size, src_len, num_heads, -1)

        return self.attention_layer(q, k, v, attn_mask)



