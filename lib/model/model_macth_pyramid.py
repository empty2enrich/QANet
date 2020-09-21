# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch
from .model_utils import Embedding

class ModelMatchPyramid(torch.nn.Module):
  def __init__(self, config):
    super(ModelMatchPyramid, self).__init__()
    self.config = config
    self.embedding = Embedding(config)