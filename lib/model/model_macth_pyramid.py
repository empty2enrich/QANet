# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch
from .model_utils import Embedding, AttentionPyramid
from ..utils import reshape_tensor, mask

class ModelMatchPyramid(torch.nn.Module):
  def __init__(self, config):
    super(ModelMatchPyramid, self).__init__()
    self.config = config
    self.embedding = Embedding(config)
    self.attention_pyramid = AttentionPyramid(config)
    self.linear = torch.nn.Linear(4, 2)

  def pointer(self, embeddings, input_mask):
    """"""
    batch_size = input_mask.shape[0]
    embeddings = reshape_tensor(embeddings, [batch_size, -1, 4])
    embeddings = self.linear(embeddings)
    # embeddings = embeddings.permute(0, 2, 1)
    # embeddings = reshape_tensor(embeddings, [batch_size, -1, 2])
    embeddings = mask(embeddings, input_mask, -2)
    start_embeddings = embeddings[:, :, 0]
    end_embeddings = embeddings[:, :, 1]
    return start_embeddings, end_embeddings

  def forward(self, input_ids, input_mask, segment_ids):
    embeddings = self.embedding(input_ids, segment_ids)
    embeddings = self.attention_pyramid(embeddings, embeddings, input_mask)
    return self.pointer(embeddings, input_mask)

