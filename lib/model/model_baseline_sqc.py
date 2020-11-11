# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch

from lib.utils import BertConfig, load_bert, mask, reshape_tensor
from lib.config import Config
from lib.model.model_base import ModelBase
from lib.model.model_utils import DepthwiseSeparableConv, Attention, Encoder, Embedding


class ModelBaseLineSQC(ModelBase):
  """
  The model of baseline.
  """

  def __init__(self, config):
    """

    Args:
      config(Config):
    """
    super(ModelBaseLineSQC, self).__init__(config)
    # self.config = config
    # embedding
    self.bert = load_bert(config.bert_config)
    self.embed_context = Embedding(self.bert,
                                   config.bert_config.max_position_embeddings,
                                   config.bert_config.hidden_size)
    self.embed_question = Embedding(self.bert, config.max_query_length,
                                    config.bert_config.hidden_size)

    # encoder
    self.attention_direction = config.direction
    self.encoder = Encoder(config.encoder_hidden_layer_number,
                           config.bert_config.hidden_size,
                           config.bert_config.max_position_embeddings,
                           config.encoder_intermediate_dim,
                           config.attention_head_num,
                           config.attention_droup_out,
                           config.attention_use_bias,
                           bi_direction_attention=config.bi_direction_attention,
                           max_query_position=config.max_query_length,
                           attention_direction=config.direction
                           )

    # pointer
    self.pointer_linear = torch.nn.Linear(config.bert_config.hidden_size, 2)
    self.query_pointor_linear = torch.nn.Linear(config.bert_config.hidden_size, int(512 * 2 / config.max_query_length))
    # self.pointer_softmax = torch.nn.Softmax(dim=-2)



  def pointer(self, embeddings, input_mask):
    """"""
    # size: batch_size, seq_length, 2
    embeddings = self.pointer_linear(embeddings)
    embeddings = mask(embeddings, input_mask, -2)
    start_embeddings = embeddings[:, :, 0].squeeze(dim=-1)
    end_embeddings = embeddings[:, :, 1].squeeze(dim=-1)
    return start_embeddings, end_embeddings

  def query_pointer(self, embeddings, input_mask):
    """"""
    # size: batch_size, seq_length, 2
    batch_size, len, dim = embeddings.shape
    embeddings = self.query_pointor_linear(embeddings)
    embeddings = reshape_tensor(embeddings, (batch_size, 512, 2))
    embeddings = mask(embeddings, input_mask, -2)
    start_embeddings = embeddings[:, :, 0].squeeze(dim=-1)
    end_embeddings = embeddings[:, :, 1].squeeze(dim=-1)
    return start_embeddings, end_embeddings

    # embeddings = self.pointer_softmax(embeddings)
    # start_softmax = embeddings[:,:,0]
    # end_softmax = embeddings[:,:,1]
    # start, end, pro = find_max_proper_batch(start_softmax, end_softmax)
    # return start, end, pro

  def forward(self, input_ids, input_mask, segment_ids,
              query_ids=None, query_mask=None, query_segment_ids=None):
    """

    Args:
      input_ids:
      input_mask:
      segment_ids:
      query_ids(): 默认 None， None： 表示 question token 与 context token
        都在 input_ids 中，否则 input_ids 只有 context token。
      query_mask:
      query_segment_ids:

    Returns:

    """
    input_embedding = self.embed_context(input_ids, segment_ids)
    query_embedding = (self.embed_question(query_ids, query_segment_ids)
                        if query_ids is not None else input_embedding)
    if self.attention_direction == "qc":
      embedding = self.encoder(query_embedding, input_embedding, query_mask, input_mask)
      start, end = self.query_pointer(embedding, query_mask)
    else:
      embedding = self.encoder(query_embedding, input_embedding, query_mask, input_mask)
      # embedding = self.encoder(input_embedding, query_embedding, input_mask, query_mask)
      start, end = self.pointer(embedding, input_mask)

    # if self.config.bi_direction_attention:
    #   start, end = self.pointer(embedding, input_mask)
    # else:
    #   start, end = self.query_pointer(embedding, input_mask)
    return start, end
