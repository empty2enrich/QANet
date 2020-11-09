#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LL
# 主要使用 BERT 输出的 CLS 项进行结果预测
# 2019

import torch

from lib.model.model_base import ModelBase
from lib.model.model_utils import Embedding
from lib.utils import load_bert

class ModelCLS(ModelBase):

  def __init__(self, config):
    super(ModelCLS, self).__init__(config)
    self.bert = load_bert(config.bert_config)
    self.embed_context = Embedding(self.bert,
                                   config.bert_config.max_position_embeddings,
                                   config.bert_config.hidden_size)
    self.embed_question = Embedding(self.bert, config.max_query_length,
                                    config.bert_config.hidden_size)
    self.pointer = torch.nn.Linear(self.config.bert_config.hidden_size * 2, 2)


  def forward(self, context_idx, context_mask, context_segements,
              query_ids, query_mask, query_segements):
    query_emb = self.embed_question(query_ids, query_segements)
    context_emb = self.embed_context(context_idx, context_segements)
    embedding = torch.cat((query_emb[:, 0], context_emb[:, 0]), dim=2)
    embedding = self.pointer(embedding)
    embedding.squeeze(1)
    start_embeddings = embedding[:, 0]
    end_embeddings = embedding[:, 1]
    return start_embeddings, end_embeddings
