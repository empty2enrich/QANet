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
  def __init__(self, in_ch, out_ch, k, dim=1, bias=True, stride=1):
    super().__init__()
    if dim == 1:
      self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch,
                                      padding=k // 2, bias=bias, stride=stride)
      self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1, padding=0, bias=bias, stride=stride)
    elif dim == 2:
      self.depthwise_conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch,
                                      padding=k // 2, bias=bias, stride=stride)
      self.pointwise_conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1, padding=0, bias=bias,
                                            stride=1)
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


class AttentionPyramid(torch.nn.Module):
  """
  构造 attention 矩阵，对 attention 矩阵使用 conv。
  """
  def __init__(self, config):
    super(AttentionPyramid, self).__init__()
    self.config = config
    self.input_len = config.bert_config.max_position_embeddings
    self.layer_num = math.ceil(math.log(self.input_len,
                                        config.pyramid_stride)) - 1
    con_list = []
    pool_list = []
    normal = []
    for i in range(self.layer_num):
      chan_in = 1 if i == 0 else config.pyramid_chan
      chan_out = config.bert_config.max_position_embeddings if i == self.layer_num - 1 else config.pyramid_chan
      con_list.append(torch.nn.Conv2d(chan_in, chan_out,
                                      config.pyramid_kernel,
                                      config.pyramid_stride,
                                      padding=config.pyramid_kernel // 2))
      # con_list.append(DepthwiseSeparableConv(chan_in,
      #                                        chan_out,
      #                                        config.pyramid_kernel,
      #                                        config.pyramid_dim,
      #                                        stride=config.pyramid_stride))
      pool_list.append(torch.nn.MaxPool2d(config.pyramid_pool_kernel,
                                          stride=1,
                                          padding=config.pyramid_pool_kernel//2))
      normal.append(torch.nn.LayerNorm(
        (math.ceil(self.input_len/ (2 ** (i + 1))),
        math.ceil(self.input_len / (2 ** (i + 1))))
      ))
    self.pools = torch.nn.ModuleList(pool_list)
    self.conv = torch.nn.ModuleList(con_list)
    self.layer_normal = torch.nn.ModuleList(normal)

    self.linear = torch.nn.Linear(128 * 64, 4) # batch_size, 64, 256, 256

  def forward(self, query_tensor, value_tensor, attention_mask=None):
    """

    Args:
      query_tensor: batch_size, len, dim
      value_tensor: batch_size, len, dim
      attention_mask: batch_size, len

    Returns:

    """
    batch_size = query_tensor.shape[0]
    cnn_datas = []
    # size: batch_size, len, len
    attention_matrix = torch.matmul(query_tensor, value_tensor.permute(0, 2, 1))
    # TODO： attention mask 用上
    attention_matrix = torch.unsqueeze(attention_matrix, 1)
    for i in range(0, 1):
      attention_matrix = self.conv[i](attention_matrix)
      attention_matrix = torch.relu(attention_matrix)
      attention_matrix = self.pools[i](attention_matrix)
      attention_matrix = torch.relu(attention_matrix)
      attention_matrix = self.layer_normal[i](attention_matrix)
      cnn_datas.append(attention_matrix)

    attention_matrix = reshape_tensor(attention_matrix, (batch_size, 512, -1))
    attention_matrix = self.linear(attention_matrix)
    # size: batch_size, length, 4
    # attention_matrix = reshape_tensor(attention_matrix, [batch_size, -1, 4])
    if self.config.visual_cnn:
      return attention_matrix, cnn_datas
    else:
      return attention_matrix


class Embedding(torch.nn.Module):
  def __init__(self, config):
    super(Embedding, self).__init__()
    self.bert = load_bert(config.bert_config)
    self.use_position_embedding = config.use_position_embedding
    self.use_conv = config.use_conv
    self.layer_normal = torch.nn.LayerNorm([config.bert_config.max_position_embeddings,
                                  config.bert_config.hidden_size])
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
    embeddings, _ = self.bert(input_ids, segment_ids)
    if self.use_position_embedding:
      embeddings = embeddings + self.position_embedding
    embeddings = self.layer_normal(embeddings)
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
    super(Encoder, self).__init__()
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
      torch.nn.LayerNorm([max_position, dim]) for _ in range(encoder_layer_num)
    ])

  def forward(self, embeddings, input_mask):
    pre_embedding = embeddings
    for index in range(self.layer_num):
      embeddings = self.attention_layer[index](embeddings, embeddings, input_mask)
      embeddings = torch.relu(self.linear_1[index](embeddings))
      embeddings = torch.relu(self.linear_2[index](embeddings))
      embeddings = torch.relu(self.linear_3[index](embeddings))
      embeddings += pre_embedding
      embeddings = self.normal[index](embeddings)
      pre_embedding = embeddings
    return embeddings