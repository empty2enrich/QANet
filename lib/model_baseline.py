# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch

from lib.utils import BertConfig

class ModelBaseLine(torch.nn.Module):
  """
  The model of baseline.
  """

  def __init__(self, bert_path, device, dropout, use_position_embedding=True,
               max_postion=cf.max_postion, pos_dim=cf.bert_dim,
               encoder_hidden_layers=cf.encoder_hidden_layers,
               encoder_intermediate_dim=cf.encoder_intermediate_dim,
               encoder_dropout_prob=cf.encoder_dropout_prob,
               attention_head_num=cf.num_heads,
               attention_probs_dropout_prob=cf.attention_probs_dropout_prob,
               attention_use_bias=cf.attention_use_bias,
               training=True,
               use_pretrained_bert=cf.use_pretrained_bert,
               use_conv=False,
               chan_in=cf.chan_in,
               chan_out=cf.chan_out,
               kernel=cf.kernel):
    """"""
    super(ModelBaseLine, self).__init__()
    self.training = training
    # embedding
    self.dim = pos_dim
    self.bert = load_bert(bert_path, device, use_pretrained_bert,
                          cf.bert_config, cf.use_segment_embedding, LocalBert)
    self.dropout = torch.nn.Dropout(dropout)
    self.layer_normal = torch.nn.LayerNorm([max_postion, pos_dim])
    self.use_position_embedding = use_position_embedding
    self.encoder_hidden_layers = encoder_hidden_layers
    if self.use_position_embedding:
      self.init_positon_embedding(max_postion, pos_dim)
    # conv
    self.use_conv = use_conv
    self.conv = DepthwiseSeparableConv(chan_in, chan_out, kernel, dim=2)

    # encoder
    self.attention_layer = torch.nn.ModuleList([
      Attention(pos_dim, attention_head_num, attention_probs_dropout_prob,
                attention_use_bias)
      for i in range(self.encoder_hidden_layers)
    ])
    self.encoder_dropout_prob = encoder_dropout_prob
    self.encoder_linear_1 = torch.nn.ModuleList(
      [torch.nn.Linear(self.dim, self.dim)
       for i in range(self.encoder_hidden_layers)])
    self.encoder_line_intermidia = torch.nn.ModuleList(
      [torch.nn.Linear(self.dim, encoder_intermediate_dim)
       for i in range(self.encoder_hidden_layers)])
    self.encoder_line_2 = torch.nn.ModuleList(
      [torch.nn.Linear(encoder_intermediate_dim, self.dim)
       for i in range(self.encoder_hidden_layers)])

    self.encoder_normal = torch.nn.ModuleList(
      [torch.nn.LayerNorm([max_postion, pos_dim]) for _ in
       range(self.encoder_hidden_layers)])

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

  def encoder(self, embeddings, input_mask):
    prelayer_output = embeddings
    for index in range(self.encoder_hidden_layers):
      # batchsize, sequence_length, posi_duim
      embeddings = self.attention_layer[index](embeddings, embeddings,
                                               input_mask)
      embeddings = self.encoder_linear_1[index](embeddings)
      embeddings = torch.relu(embeddings)
      embeddings = self.encoder_line_intermidia[index](embeddings)
      # embeddings = gelu(embeddings)
      embeddings = torch.relu(embeddings)
      embeddings = self.encoder_line_2[index](embeddings)
      embeddings = torch.relu(embeddings)
      embeddings += prelayer_output
      # todo: dropout、 normal
      embeddings = self.encoder_normal[index](embeddings)
      # embeddings = functional.leaky_relu(embeddings)
      # embeddings = functional.dropout(embeddings, self.encoder_dropout_prob, self.training)
      prelayer_output = embeddings
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
