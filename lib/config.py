# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

from lib.utils import BertConfig

class Config(object):
  def __init__(self):
    self.device = "cpu"
    self.dropout = 0.1
    # encoder
    self.use_position_embedding = True
    self.encoder_hidden_layer_number = 11
    self.encoder_intermediate_dim = 3072
    self.encoder_dropout = 0.1
    # attention
    self.attention_head_num = 16
    self.attention_droup_out = 0.1
    self.attention_use_bias = False
    # train
    self.training = True
    # cnn
    self.use_conv=False
    self.chan_in = 768
    self.chan_out = 768
    self.kernel = 7
    self.dim = 2
    # bert config
    self.bert_config = BertConfig()

  def load_from_json(self, json_path):
    """"""