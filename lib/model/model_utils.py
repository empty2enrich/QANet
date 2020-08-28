# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import math
import torch
from lib.utils import reshape_tensor, load_bert

class DepthwiseSeparableConv(torch.nn.Module):
  def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
    super().__init__()
    if dim == 1:
      self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                      padding=k // 2, bias=bias)
      self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    elif dim == 2:
      self.depthwise_conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                      padding=k // 2, bias=bias)
      self.pointwise_conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    else:
      raise Exception("Wrong dimension for Depthwise Separable Convolution!")
    # torch.nn.init.kaiming_normal_(self.depthwise_conv.weight)
    # torch.nn.init.constant_(self.depthwise_conv.bias, 0.0)
    # torch.nn.init.kaiming_normal_(self.depthwise_conv.weight)
    # torch.nn.init.constant_(self.pointwise_conv.bias, 0.0)

  def forward(self, x):
    return self.pointwise_conv(self.depthwise_conv(x))


class Attention(torch.nn.Module):
  """
  Attention
  """
  def __init__(self, dim, attention_head_num, attention_probs_dropout_prob,
              use_bias=False):
    super(Attention, self).__init__()
    self.dim = dim
    self.attention_head_num = attention_head_num
    self.use_bias = use_bias
    self.dropout = torch.nn.Dropout(attention_probs_dropout_prob)
    if not self.dim % self.attention_head_num == 0:
      raise Exception(f"The dim({self.dim}) % attention_head_num({self.attention_head_num}) != 0")
    self.size_per_head = int(self.dim / self.attention_head_num)
    self.query_layer = torch.nn.Linear(self.dim, self.dim, self.use_bias)
    self.key_layer = torch.nn.Linear(self.dim, self.dim, self.use_bias)
    self.value_layer = torch.nn.Linear(self.dim, self.dim, self.use_bias)
    self.softmax = torch.nn.Softmax(dim=-1)

  def transpose4score(self, tensor, shape):
    """
    为计算 score 对 tensor 进行转换.
    Args:
      tensor:
      shape:

    Returns:

    """
    tensor = reshape_tensor(tensor, shape)
    tensor = tensor.permute(0, 2, 1, 3)
    return tensor

  def forward(self, query_tensor, value_tensor, attention_mask=None):
    """"""
    batch_size, quert_length, _ = query_tensor.shape
    _, value_length, _ = value_tensor.shape

    query_tensor = reshape_tensor(query_tensor, (-1, self.dim))
    value_tensor = reshape_tensor(value_tensor, (-1, self.dim))
    query_tensor = self.query_layer(query_tensor)
    key_tensor = self.key_layer(value_tensor)
    value_tensor = self.value_layer(value_tensor)

    query_tensor = self.transpose4score(query_tensor, (batch_size, quert_length,
                                                       self.attention_head_num,
                                                       self.size_per_head))
    key_tensor = self.transpose4score(key_tensor, (batch_size, value_length,
                                                   self.attention_head_num,
                                                   self.size_per_head))
    attention_scores = torch.matmul(query_tensor, key_tensor.permute(0, 1, 3, 2))
    # batch_size, attention_head_num, query_length, value_length
    attention_scores = attention_scores / math.sqrt(float(self.size_per_head))

    if attention_mask is not None:
      # batch_size, 1, sqe_len
      attention_mask = torch.unsqueeze(attention_mask, 1)
      # batch_size, 1, sqe_len, 1
      attention_mask = torch.unsqueeze(attention_mask, -1)
      # batch_size, attention_head_num, squ_len
      attention_mask = attention_mask.expand(batch_size, self.attention_head_num, quert_length, value_length)
      attention_scores = attention_scores * attention_mask

    attention_scores = self.softmax(attention_scores)
    # attention_scores = self.dropout(attention_scores)

    value_tensor = reshape_tensor(value_tensor, (batch_size, value_length,
                                                 self.attention_head_num, self.size_per_head))

    value_tensor = value_tensor.permute(0, 2, 1, 3)
    # batch_size, attention_head_num, query_len, size_per_head
    attention = torch.matmul(attention_scores, value_tensor)

    # batch_size, attention_head_num, query_length, size_per_head
    # attention = torch.matmul(attention_mask, value_tensor)

    attention = attention.permute(0, 2, 1, 3)
    attention = reshape_tensor(attention, (batch_size, quert_length, self.dim))

    return attention


class Embedding(torch.nn.Module):
  def __init__(self, config):
    self.bert = load_bert(config.bert_config)
    self.use_position_embedding = config.use_position_embedding
    self.use_conv = config.use_conv
    if self.use_position_embedding:
      self.init_positon_embedding(config.bert_config.max_position_embeddings,
                                  config.bert_config.hidden_size)

    self.conv = DepthwiseSeparableConv(config.chan_in, config.chan_out,
                                       config.kernel, config.dim)

  def init_positon_embedding(self, max_postion, pos_dim):
    posi_embedding = torch.Tensor(max_postion, pos_dim)
    # posi_embedding = torch.nn.init.kaiming_normal(posi_embedding, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
    self.position_embedding = torch.nn.Parameter(posi_embedding)
    torch.nn.init.kaiming_normal_(self.position_embedding, mode='fan_out')

  def forward(self, input_ids, segment_ids):
    """"""
    embeddings = self.bert(input_ids, segment_ids)
    if self.use_position_embedding:
      embeddings = embeddings + self.position_embedding
    if self.use_conv:
      embeddings = embeddings.unsqueeze(-1)
      embeddings = embeddings.permute(0, 3, 2, 1)
      embeddings = self.conv(embeddings)
      embeddings = embeddings.permute(0, 3, 2, 1)
      embeddings = embeddings.squeeze(-1)
    return embeddings

class Encoder(torch.nn.Module):
  """
  对词向量进行编码
  """

  def __init__(self, encoder_layer_num, dim, max_position, intermediate_dim, attention_head_num,
               attention_pro_drop, attention_use_bias=False):
    self.layer_num = encoder_layer_num
    self.attention_layer = torch.nn.ModuleList([
      Attention(dim, attention_head_num, attention_pro_drop, attention_use_bias)
      for i in range(encoder_layer_num)
    ])
    self.linear_1 = torch.nn.ModuleList([
      torch.nn.Linear(dim, dim) for i in range(encoder_layer_num)
    ])
    self.linear_2 = torch.nn.ModuleList([
      torch.nn.Linear(dim, intermediate_dim) for i in range(encoder_layer_num)
    ])
    self.linear_3 = torch.nn.ModuleList([
      torch.nn.Linear(intermediate_dim, dim) for i in range(encoder_layer_num)
    ])
    self.normal = torch.nn.ModuleList([
      torch.nn.LayerNorm([max_position, dim]) for _ in encoder_layer_num
    ])

  def forward(self, embeddings, input_mask):
    pre_embedding = embeddings
    for index in range(self.layer_num):
      embeddings = self.attention_layer[index](embeddings, embeddings, input_mask)
      embeddings = torch.relu(self.linear_1[index](embeddings))
      embeddings = torch.relu(self.linear_2[index](embeddings))
      embeddings = torch.relu(self.linear_3[index](embeddings))
      embeddings += embeddings
      embeddings = self.normal[index](embeddings)
      pre_embedding = embeddings
    return embeddings