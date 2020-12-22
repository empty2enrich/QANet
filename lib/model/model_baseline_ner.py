# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch

from lib.utils import BertConfig, load_bert, mask
from lib.config import Config
from lib.model.model_base import ModelBase
from lib.model.model_utils import DepthwiseSeparableConv, Attention, Encoder, Embedding, CRF

class ModelBaseLineNER(ModelBase):
  """
  The model of baseline. binary classify
  """

  def __init__(self, config):
    """

    Args:
      config(Config):
    """
    super(ModelBaseLineNER, self).__init__(config)
    # self.config = config
    # embedding
    self.bert = load_bert(config.bert_config)
    self.embed_word = Embedding(self.bert,
                                   config.bert_config.max_position_embeddings,
                                   config.bert_config.hidden_size)

    # encoder
    self.encoder = Encoder(config.encoder_hidden_layer_number,
                           config.bert_config.hidden_size,
                           config.bert_config.max_position_embeddings,
                           config.encoder_intermediate_dim,
                           config.attention_head_num,
                           config.attention_droup_out,
                           config.attention_use_bias)

    self.rnn = torch.nn.LSTM(config.bert_config.hidden_size,
                             config.lstm_hidden_size,
                             num_layers=config.lstm_layer_num,
                             batch_first=config.lstm_batch_first,
                             bidirectional=config.lstm_bi_direction)
    self.rnn_linear = torch.nn.Linear(self.config.lstm_hidden_size * (2 if self.config.lstm_bi_direction else 1),
                                      self.config.crf_target_size + 2)
    self.rnn_noraml = torch.nn.LayerNorm((self.config.bert_config.max_position_embeddings,
                                          self.config.crf_target_size + 2))

    self.crf = CRF(self.config.crf_target_size, self.config.device=="cuda", self.config.crf_average_batch)



    # pointer
    self.pointer_linear = torch.nn.Linear(config.bert_config.hidden_size, self.config.crf_target_size + 2)
    # self.pointer_softmax = torch.nn.Softmax(dim=-2)

  def init_h0_c0(self):
    """"""
    h0 = torch.Tensor(self.config.lstm_layer_num * (2 if self.config.lstm_bi_direction else 1),
                      self.config.batch_size,
                      self.config.bert_config.hidden_size)
    c0 = torch.Tensor(self.config.lstm_layer_num * (2 if self.config.lstm_bi_direction else 1),
                      self.config.batch_size,
                      self.config.bert_config.hidden_size)
    h0 = torch.nn.init.kaiming_normal_(h0).to(self.config.device)
    c0 = torch.nn.init.kaiming_normal_(c0).to(self.config.device)
    return h0, c0

  def pointer(self, embeddings, input_mask):
    """"""
    # size: batch_size, seq_length, 2
    embeddings = self.pointer_linear(embeddings)
    embeddings = mask(embeddings, input_mask, -2)
    # start_embeddings = embeddings[:, :, 0].squeeze(dim=-1)
    # end_embeddings = embeddings[:, :, 1].squeeze(dim=-1)
    return embeddings

    # embeddings = self.pointer_softmax(embeddings)
    # start_softmax = embeddings[:,:,0]
    # end_softmax = embeddings[:,:,1]
    # start, end, pro = find_max_proper_batch(start_softmax, end_softmax)
    # return start, end, pro

  def forward(self, input_ids, input_mask, segment_ids):
    embedding = self.embed_word(input_ids, segment_ids)
    if self.config.use_lstm:
      embedding, _ = self.rnn(embedding.permute(1, 0, 2), self.init_h0_c0())
      embedding = torch.relu(self.rnn_linear(embedding.permute(1, 0, 2)))
      embedding = self.rnn_noraml(embedding)
    if self.config.use_encoder:
      embedding = self.encoder(embedding, input_mask)
      embedding = self.pointer(embedding, input_mask)
    return embedding

  def loss(self, feats, mask, tags):
    """"""
    loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
    batch_size = feats.size(0)
    loss_value /= float(batch_size)
    return loss_value