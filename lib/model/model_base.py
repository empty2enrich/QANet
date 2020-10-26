# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch

class ModelBase(torch.nn.Module):
  def __init__(self, config):
    """

    Args:
      config:
    """
    super(ModelBase, self).__init__()
    self.config = config
    self.additional_datas = []

  def get_additional_datas(self):
    return self.additional_datas