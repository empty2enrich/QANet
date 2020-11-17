# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import math
import torch
from lib.utils import reshape_tensor, load_bert, multiply
from torch.autograd import Variable

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

  def forward(self, query_tensor, value_tensor, value_attention_mask=None):
    """
    输出的 length 与 query_tensor  length 保持一致。
    Args:
      query_tensor:
      value_tensor:
      value_attention_mask:

    Returns:

    """
    batch_size, query_length, _ = query_tensor.shape
    _, value_length, _ = value_tensor.shape

    query_tensor = reshape_tensor(query_tensor, (-1, self.dim))
    value_tensor = reshape_tensor(value_tensor, (-1, self.dim))
    query_tensor = self.query_layer(query_tensor)
    key_tensor = self.key_layer(value_tensor)
    value_tensor = self.value_layer(value_tensor)

    query_tensor = self.transpose4score(query_tensor, (batch_size, query_length,
                                                       self.attention_head_num,
                                                       self.size_per_head))
    key_tensor = self.transpose4score(key_tensor, (batch_size, value_length,
                                                   self.attention_head_num,
                                                   self.size_per_head))
    attention_scores = torch.matmul(query_tensor, key_tensor.permute(0, 1, 3, 2))
    # batch_size, attention_head_num, query_length, value_length
    attention_scores = attention_scores / math.sqrt(float(self.size_per_head))

    if value_attention_mask is not None:
      # batch_size, 1, sqe_len
      value_attention_mask = torch.unsqueeze(value_attention_mask, 1)
      # batch_size, 1, sqe_len, 1
      value_attention_mask = torch.unsqueeze(value_attention_mask, -1)
      # batch_size, attention_head_num, squ_len
      value_attention_mask = value_attention_mask.expand(batch_size, self.attention_head_num, query_length, value_length)
      attention_scores = attention_scores * value_attention_mask

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
    attention = reshape_tensor(attention, (batch_size, query_length, self.dim))

    return attention


class AttentionPyramid(torch.nn.Module):
  """
  构造 attention 矩阵，对 attention 矩阵使用 conv。
  """
  def __init__(self, config):
    super(AttentionPyramid, self).__init__()
    self.config = config
    self.input_len = config.bert_config.max_position_embeddings
    # self.layer_num = math.ceil(math.log(self.input_len,
    #                                     config.pyramid_pool_stride)) - 1
    self.layer_num = math.ceil(math.log(self.input_len,
                                        config.pyramid_pool_stride)) - 4

    # self.layer_num = 3 # TODO: 先试试 3
    con_list = []
    pool_list = []
    adptors = []
    normal = []
    chan_in = 1
    chan_out = config.pyramid_chan
    for i in range(self.layer_num):
      con_list.append(torch.nn.Conv2d(chan_in, chan_out,
                                      config.pyramid_kernel,
                                      config.pyramid_stride,
                                      padding=config.pyramid_kernel // 2))
      con_list.append(torch.nn.Conv2d(chan_out, chan_out,
                                      config.pyramid_kernel,
                                      config.pyramid_stride,
                                      padding=config.pyramid_kernel // 2))
      con_list.append(torch.nn.Conv2d(chan_out, chan_out,
                                      config.pyramid_kernel,
                                      config.pyramid_stride,
                                      padding=config.pyramid_kernel // 2))
      # con_list.append(DepthwiseSeparableConv(chan_in,
      #                                        chan_out,
      #                                        config.pyramid_kernel,
      #                                        config.pyramid_dim,
      #                                        stride=config.pyramid_stride))
      pool_list.append(torch.nn.MaxPool2d(config.pyramid_pool_kernel,
                                          stride=config.pyramid_pool_stride,
                                          # padding=config.pyramid_pool_kernel//2))
                                          padding=0))
      cur_len = math.ceil(self.input_len/ (2 ** (i + 1)))
      normal.append(torch.nn.LayerNorm((cur_len, cur_len)))


      adptors.append(ResNetCNNAdaptor((1, self.input_len, self.input_len),
                                      (chan_out, cur_len, cur_len)))


      chan_in = chan_out
      chan_out *= 2
    self.pools = torch.nn.ModuleList(pool_list)
    self.conv = torch.nn.ModuleList(con_list)
    self.layer_normal = torch.nn.ModuleList(normal)
    self.adaptor = torch.nn.ModuleList(adptors)

    # (512 / 2 ** layer_num) ** 2 * 64 * 2 **(layer_num -1) / 512

    self.linear = torch.nn.Linear(self.get_linear_input_feature(), 4)

  def get_linear_input_feature(self):
    return int((512 / 2 ** self.layer_num) ** 2 * 64 * 2 **(self.layer_num -1) / 512)

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
    origin_matrix = attention_matrix
    for i in range(0, self.layer_num):
      attention_matrix = self.conv[i * 3](attention_matrix)
      attention_matrix = self.conv[i * 3 + 1](attention_matrix)
      attention_matrix = self.conv[i * 3 + 2](attention_matrix)
      attention_matrix = torch.relu(attention_matrix) # todo: 测试下有、没有性能一样不
      attention_matrix = self.pools[i](attention_matrix)
      attention_matrix = torch.relu(attention_matrix)
      attention_matrix += self.adaptor[i](origin_matrix)
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
  """
  将 token 转为 embeddings.
  """
  def __init__(self, bert, length, dim, use_position_embedding=True,
               use_conv=False, chan_in=None, chan_out=None, kernel=3, cnn_dim=2):
    super(Embedding, self).__init__()
    self.bert = bert
    self.use_position_embedding = use_position_embedding
    self.use_conv = use_conv
    self.layer_normal = torch.nn.LayerNorm([length, dim])
    if self.use_position_embedding:
      self.init_positon_embedding(length, dim)
    if use_conv:
      self.conv = DepthwiseSeparableConv(chan_in, chan_out, kernel, cnn_dim)

  def init_positon_embedding(self, max_postion, pos_dim):
    """
    初始化 position embeddings.
    Args:
      max_postion:
      pos_dim:

    Returns:

    """
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


class EncoderSQC(torch.nn.Module):
  """
  对词向量进行编码, question、context token 分开。
  """

  def __init__(self, encoder_layer_num, dim, max_context_position, intermediate_dim, attention_head_num,
               attention_pro_drop, attention_use_bias=False, bi_direction_attention=False, max_query_position=32,
               attention_direction="qc"):
    super(EncoderSQC, self).__init__()
    self.layer_num = encoder_layer_num
    self.bi_direction_attention = bi_direction_attention
    self.attention_direction = attention_direction
    # attention: query -> context
    self.attention_layer_qc = torch.nn.ModuleList([
      Attention(dim, attention_head_num, attention_pro_drop, attention_use_bias)
      for i in range(encoder_layer_num)
    ])
    # attention: context -> question
    self.attention_layer_cq = torch.nn.ModuleList([
      Attention(dim, attention_head_num, attention_pro_drop, attention_use_bias)
      for i in range(encoder_layer_num)
    ])
    self.linear_1_qc = torch.nn.ModuleList([
      torch.nn.Linear(dim, dim) for i in range(encoder_layer_num)
    ])
    self.linear_1_cq = torch.nn.ModuleList([
      torch.nn.Linear(dim, dim) for i in range(encoder_layer_num)
    ])
    self.linear_2_qc = torch.nn.ModuleList([
      torch.nn.Linear(dim, intermediate_dim) for i in range(encoder_layer_num)
    ])
    self.linear_2_cq = torch.nn.ModuleList([
      torch.nn.Linear(dim, intermediate_dim) for i in range(encoder_layer_num)
    ])
    self.linear_3_qc = torch.nn.ModuleList([
      torch.nn.Linear(intermediate_dim, dim) for i in range(encoder_layer_num)
    ])
    self.linear_3_cq = torch.nn.ModuleList([
      torch.nn.Linear(intermediate_dim, dim) for i in range(encoder_layer_num)
    ])
    self.normal_qc = torch.nn.ModuleList([
      torch.nn.LayerNorm([max_query_position, dim]) for _ in range(encoder_layer_num)
    ])
    self.normal_cq = torch.nn.ModuleList([
      torch.nn.LayerNorm([max_context_position, dim]) for _ in range(encoder_layer_num)
    ])

  def encode_qc(self, query_embeddings, context_embeddings, query_mask, layer_num, pre_query_embedding):
    """
    query -> context attention encode。
    Args:
      query_embeddings:
      context_embeddings:
      query_mask:
      context_mask:
      layer_num:
      pre_query_embedding:

    Returns:

    """
    query_embeddings = self.attention_layer_qc[layer_num](query_embeddings, context_embeddings, query_mask)
    query_embeddings = torch.relu(self.linear_1_qc[layer_num](query_embeddings))
    query_embeddings = torch.relu(self.linear_2_qc[layer_num](query_embeddings))
    query_embeddings = torch.relu(self.linear_3_qc[layer_num](query_embeddings))
    query_embeddings += pre_query_embedding
    query_embeddings = self.normal_qc[layer_num](query_embeddings)
    return query_embeddings

  def encode_cq(self, query_embeddings, context_embeddings, context_mask, layer_num, pre_context_embedding):
    """
    context -> query attention encode。
    Args:
      query_embeddings:
      context_embeddings:
      query_mask:
      context_mask:
      layer_num:
      pre_context_embedding:

    Returns:

    """
    context_embeddings = self.attention_layer_cq[layer_num](context_embeddings, query_embeddings, context_mask)
    context_embeddings = torch.relu(self.linear_1_cq[layer_num](context_embeddings))
    context_embeddings = torch.relu(self.linear_2_cq[layer_num](context_embeddings))
    context_embeddings = torch.relu(self.linear_3_cq[layer_num](context_embeddings))
    context_embeddings += pre_context_embedding
    context_embeddings = self.normal_cq[layer_num](context_embeddings)
    return context_embeddings

  def forward(self, query_embeddings, context_embeddings, query_mask, context_mask=None):
    pre_embedding = query_embeddings
    pre_context_embeddings = context_embeddings
    for index in range(self.layer_num):
      # query attention
      if self.attention_direction == "qc":
        query_embeddings = self.encode_qc(query_embeddings, context_embeddings, query_mask, index, pre_embedding)
        # context attention
        if self.bi_direction_attention:
          context_embeddings = self.encode_cq(query_embeddings, context_embeddings, context_mask, index, pre_context_embeddings)
      elif self.attention_direction == "cq":
        context_embeddings = self.encode_cq(query_embeddings, context_embeddings, context_mask, index,
                                            pre_context_embeddings)
        if self.bi_direction_attention:
          query_embeddings = self.encode_qc(query_embeddings, context_embeddings, query_mask, index, pre_embedding)

      pre_embedding = query_embeddings
      pre_context_embeddings = context_embeddings
    if self.attention_direction == "qc":
      return query_embeddings
    else:
      return context_embeddings


################################################################################
################################# RNN #############################################
################################################################################

class ResNetCNNAdaptor(torch.nn.Module):
  """
  CNN 的 ResNet 适配器
  cnn 输入与深层的 cnn 输入 shape 不一致，无法直接相加，使用 linear 修改输入 shape。
  """
  def __init__(self, input_shape, output_shape):
    super(ResNetCNNAdaptor, self).__init__()
    self.input_shape = list(input_shape)
    self.output_shape = list(output_shape)
    scale = multiply(*self.input_shape) / multiply(*self.output_shape)
    self.conv = torch.nn.Conv2d(self.input_shape[0], int(self.input_shape[0] / scale), kernel_size=1)
    # self.linear = torch.nn.Linear(self.input_shape[2],
    #                               int(self.input_shape[2] / scale))

  def forward(self, tensor):
    batch_size = tensor.shape[0]
    tensor = self.conv(tensor)
    # tensor = reshape_tensor(tensor, (batch_size, self.output_shape[0], -1))
    # tensor = self.linear(tensor)
    tensor = reshape_tensor(tensor, [batch_size] + self.output_shape)
    return tensor



class CRF(torch.nn.Module):
  """
  CRF model.
  """
  def __init__(self, target_size, average_batch=True):
    """
    Args:
      target_size(int): Target size.
      use_cuda(bool): Whether use GPU.
      average_batch(bool): Whether loss user average.
    """
    super(CRF, self).__init__()
    # for k in kwargs:
    #     self.__setattr__(k, kwargs[k])
    self.target_size = target_size
    self.average_batch = average_batch
    self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
    init_transitions = torch.zeros(self.target_size+2, self.target_size+2)
    init_transitions[:, self.START_TAG_IDX] = -1000.
    init_transitions[self.END_TAG_IDX, :] = -1000.
    # if self.use_cuda:
    #   self.device = torch.device(self.devices[0])
    #   init_transitions = init_transitions.to(self.device)
    self.transitions = torch.nn.Parameter(init_transitions)

  def _forward_alg(self, feats, mask=None):
    """
    Do the forward algorithm to compute the partition function (batched).

    Args:
      feats(Tensor): Tensor of feats, eg:(batch_size, seq_len, self.target_size+2).
      mask(Tensor): Tensor of mask, eg:(batch_size, seq_len).

    Returns:
      final_partition(float): Num of final partition.
      scores(float): Num of score.
    """
    batch_size = feats.size(0)
    seq_len = feats.size(1)
    tag_size = feats.size(-1)

    mask = mask.transpose(1, 0).contiguous()
    ins_num = batch_size * seq_len
    feats = feats.transpose(1, 0).contiguous().view(
        ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

    scores = feats + self.transitions.view(
        1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
    scores = scores.view(seq_len, batch_size, tag_size, tag_size)
    seq_iter = enumerate(scores)
    try:
      _, inivalues = seq_iter.__next__()
    except:
      _, inivalues = seq_iter.next()

    partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
    for idx, cur_values in seq_iter:
      cur_values = cur_values + partition.contiguous().view(
          batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
      cur_partition = log_sum_exp(cur_values, tag_size)
      mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
      masked_cur_partition = cur_partition.masked_select(mask_idx.byte())
      if masked_cur_partition.dim() != 0:
        mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
        partition.masked_scatter_(mask_idx.byte(), masked_cur_partition)
    cur_values = self.transitions.view(1, tag_size, tag_size).expand(
        batch_size, tag_size, tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
    cur_partition = log_sum_exp(cur_values, tag_size)
    final_partition = cur_partition[:, self.END_TAG_IDX]
    return final_partition.sum(), scores

  def _viterbi_decode(self, feats, mask=None):
    """
    Viterbi decode.
    Args:
      feats(Tensor): Tensor of feats, eg:(batch_size, seq_len, self.target_size+2).
      mask(Tensor): Tensor of mask, eg:(batch_size, seq_len).

    Returns:
      decode_idx: Result of viterbi decode, eg:(batch_size, seq_len).
      path_score(float): Score of every sentence.
    """
    batch_size = feats.size(0)
    seq_len = feats.size(1)
    tag_size = feats.size(-1)

    length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
    mask = mask.transpose(1, 0).contiguous()
    ins_num = seq_len * batch_size
    feats = feats.transpose(1, 0).contiguous().view(
      ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

    scores = feats + self.transitions.view(
      1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
    scores = scores.view(seq_len, batch_size, tag_size, tag_size)

    seq_iter = enumerate(scores)
    # record the position of the best score
    back_points = list()
    partition_history = list()
    mask = (1 - mask.long()).byte()
    try:
      _, inivalues = seq_iter.__next__()
    except:
      _, inivalues = seq_iter.next()
    partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
    partition_history.append(partition)

    for idx, cur_values in seq_iter:
      cur_values = cur_values + partition.contiguous().view(
          batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
      partition, cur_bp = torch.max(cur_values, 1)
      partition_history.append(partition.unsqueeze(-1))
      cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
      back_points.append(cur_bp)

    partition_history = torch.cat(partition_history).view(
        seq_len, batch_size, -1).transpose(1, 0).contiguous()

    last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
    last_partition = torch.gather(
        partition_history, 1, last_position).view(batch_size, tag_size, 1)

    last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
        self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
    _, last_bp = torch.max(last_values, 1)
    pad_zero = Variable(torch.zeros(batch_size, tag_size)).long()
    if self.use_cuda:
      pad_zero = pad_zero.cuda()
    back_points.append(pad_zero)
    back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

    pointer = last_bp[:, self.END_TAG_IDX]
    insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
    back_points = back_points.transpose(1, 0).contiguous()

    back_points.scatter_(1, last_position, insert_last)

    back_points = back_points.transpose(1, 0).contiguous()

    decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
    if self.use_cuda:
      decode_idx = decode_idx.cuda()
    decode_idx[-1] = pointer.data
    for idx in range(len(back_points)-2, -1, -1):
      pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
      decode_idx[idx] = pointer.view(-1).data
    path_score = None
    decode_idx = decode_idx.transpose(1, 0)
    return path_score, decode_idx

  def forward(self, feats, mask=None):
    """
    Forward compute.
    Args:
      feats(Tensor): Tensor of feats, eg:(batch_size, seq_len, self.target_size+2).
      mask(Tensor): Tensor of mask, eg:(batch_size, seq_len).

    Returns:
      decode_idx: Result of viterbi decode, eg:(batch_size, seq_len).
      path_score(float): Score of every sentence.
    """
    path_score, best_path = self._viterbi_decode(feats, mask)
    return path_score, best_path

  def _score_sentence(self, scores, mask, tags):
    """
    Args:
      scores(Tensor): Score of sentense, eg:(seq_len, batch_size, tag_size, tag_size).
      mask(Tensor): Tensor of mask, eg:(batch_size, seq_len).
      tags(Tensor): Tensor of tags, eg:(batch_size, seq_len).

    Returns:
      score(float): Gold score.
    """
    batch_size = scores.size(1)
    seq_len = scores.size(0)
    tag_size = scores.size(-1)

    new_tags = Variable(torch.LongTensor(batch_size, seq_len))
    if self.use_cuda:
        new_tags = new_tags.cuda()
    for idx in range(seq_len):
      if idx == 0:
        new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
      else:
        new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]
    end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
        1, tag_size).expand(batch_size, tag_size)
    length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
    end_ids = torch.gather(tags, 1, length_mask-1)
    end_energy = torch.gather(end_transition, 1, end_ids)
    new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
    tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
        seq_len, batch_size)
    tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
    gold_score = tg_energy.sum() + end_energy.sum()
    return gold_score

  def neg_log_likelihood_loss(self, feats, mask, tags):
    """
    Compute logarithm likelihood loss.
    Args:
      feats(Tensor): Tensor of feats, eg:(batch_size, seq_len, self.target_size+2).
      mask(Tensor): Tensor of mask, eg:(batch_size, seq_len).
      tags: Tensor of tags, eg:(batch_size, seq_len).
    """
    batch_size = feats.size(0)
    mask = mask.byte()
    forward_score, scores = self._forward_alg(feats, mask)
    gold_score = self._score_sentence(scores, mask, tags)
    if self.average_batch:
        return (forward_score - gold_score) / batch_size
    return forward_score - gold_score


def log_sum_exp(vec, m_size):
  """
  Args:
    vec(Tensor): High dimention tensor, eg:(batch_size, vanishing_dim, hidden_dim).
    m_size(int): Hidden dimention.

  Returns:
    size(Tensor): Tensor, eg:(batch_size, hidden_dim).
  """
  _, idx = torch.max(vec, 1)  # B * 1 * M
  max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
  return max_score.view(-1, m_size) + torch.log(torch.sum(
    torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)