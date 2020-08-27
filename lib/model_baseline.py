# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch

from lib.utils import BertConfig, load_bert
from lib.config import Config
from lib.model_utils import DepthwiseSeparableConv, Attention, Encoder

class ModelBaseLine(torch.nn.Module):
  """
  The model of baseline.
  """

  def __init__(self, config):
    """

    Args:
      config(Config):
    """
    super(ModelBaseLine, self).__init__()
    self.config = config
    # embedding
    # self.dim = pos_dim
    self.bert = load_bert(config.bert_config)
    self.dropout = torch.nn.Dropout(config.dropout)
    self.layer_normal = torch.nn.LayerNorm([config.bert_config.max_position_embeddings,
                                            config.bert_config.hidden_size])
    # self.use_position_embedding = use_position_embedding
    # self.encoder_hidden_layers = encoder_hidden_layers
    if self.config.use_position_embedding:
      self.init_positon_embedding(config.bert_config.max_position_embeddings,
                                  config.bert_config.hidden_size)
    # conv
    self.conv = DepthwiseSeparableConv(config.chan_in, config.chan_out,
                                       config.kernel, config.dim)

    # encoder
    self.encoder = Encoder(config.encoder_hidden_layer_number,
                           config.bert_config.hidden_size,
                           config.bert_config.max_position_embeddings,
                           config.encoder_intermediate_dim,
                           config.attention_head_num,
                           config.attention_droup_out,
                           config.attention_use_bias)

    # pointer
    self.pointer_linear = torch.nn.Linear(self.dim, 2)
    # self.pointer_softmax = torch.nn.Softmax(dim=-2)

  def init_positon_embedding(self, max_postion, pos_dim):
    posi_embedding = torch.Tensor(max_postion, pos_dim)
    # posi_embedding = torch.nn.init.kaiming_normal(posi_embedding, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
    self.position_embedding = torch.nn.Parameter(posi_embedding)
    torch.nn.init.kaiming_normal_(self.position_embedding, mode='fan_out')

  def embedding(self, input_ids, segment_ids):
    """
    Embedding for input.
    Args:
      input_ids:
      segment_ids:

    Returns:

    """
    embeddings, _ = self.bert(input_ids, segment_ids)
    if self.use_position_embedding:
      embeddings = embeddings + self.position_embedding
    # batch_size, length, dim
    embeddings = self.layer_normal(embeddings)
    if self.use_conv:
      embeddings = embeddings.unsqueeze(-1)
      embeddings = embeddings.permute(0, 3, 2, 1)
      embeddings = self.conv(embeddings)
      embeddings = embeddings.permute(0, 3, 2, 1)
      embeddings = embeddings.squeeze(-1)
    # embeddings = self.dropout(embeddings)
    return embeddings


  def pointer(self, embeddings, input_mask):
    """"""
    # size: batch_size, seq_length, 2
    embeddings = self.pointer_linear(embeddings)
    embeddings = mask(embeddings, input_mask, -2)
    start_embeddings = embeddings[:, :, 0].squeeze(dim=-1)
    end_embeddings = embeddings[:, :, 1].squeeze(dim=-1)
    return start_embeddings, end_embeddings

    # embeddings = self.pointer_softmax(embeddings)
    # start_softmax = embeddings[:,:,0]
    # end_softmax = embeddings[:,:,1]
    # start, end, pro = find_max_proper_batch(start_softmax, end_softmax)
    # return start, end, pro

  def forward(self, input_ids, input_mask, segment_ids):
    embedding = self.embedding(input_ids, segment_ids)
    embedding = self.encoder(embedding, input_mask)
    start, end = self.pointer(embedding, input_mask)
    return start, end
