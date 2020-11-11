#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LL
# 2019

import torch
from lib.model.model_base import ModelBase

class ModelLSTM(ModelBase):
  """"""
  def __init__(self, config):
    super(ModelLSTM, self).__init__(config)
    self.encode = torch.nn.LSTM()

