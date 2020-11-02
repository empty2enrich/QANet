# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import copy
import json
import os
import torch
import six

import pytorch_transformers
from my_py_toolkit.file.file_toolkit import readjson

class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size=None,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               device="cuda",
               use_pretrained_bert=False,
               use_segment_embedding=True,
               bert_path=None,
               vocab_file=None,
               bert_config_path=None,
               bert_class=None,
               freeze_paras=False):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      freeze_paras(bool): 是否冻结 bert 的参数。True 冻结。
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

    self.device = device
    self.use_pretrained_bert = use_pretrained_bert
    self.use_segment_embedding = use_segment_embedding

    self.bert_path = bert_path
    self.vocab_file = vocab_file
    self.bert_config_path = bert_config_path
    self.bert_class = bert_class
    self.freeze_paras = freeze_paras

    # 根据 bert path 初始化 config, vocab 等信息
    self.init_with_bert_path()

  def init_with_bert_path(self):
    if not self.bert_path:
      return
    if not self.vocab_file:
      self.vocab_file = os.path.join(self.bert_path, "vocab.txt")
    if not self.bert_config_path:
      self.bert_config_path = os.path.join(self.bert_path, "bert_config.json")
    if not self.vocab_size:
      with open(self.vocab_file, "r", encoding="utf-8") as f:
        vocab = f.read()
        self.vocab_size = len(vocab.split("\n"))

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    return cls.from_dict(readjson(json_file))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def load_bert(bert_config):
  """

  Args:
    bert_config(BertConfig):
    freeze_paras(bool): 是否冻结参数， True 冻结。
  Returns:

  """
  if bert_config.use_pretrained_bert:
    model = bert_config.bert_class.from_pretrained(bert_config.bert_path).to(bert_config.device)
  else:
    model = bert_config.bert_class(pytorch_transformers.BertConfig.from_dict(bert_config.to_dict())).to(bert_config.device)

  if bert_config.freeze_paras:
    for _, param in model.named_parameters():
      param.requires_grad_(False)
  return model

def reshape_tensor(tensor, shape):
  return tensor.contiguous().view(*shape)


def mask(tensor, tensor_mask, mask_dim):
  """
  Mask a tensor.
  Args:
    tensor(torch.Tensor): 输入
    tensor_mask(torch.Tensor): mask 位置信息.
    mask_dim(int): 负数，指定需要 mask 的维度，example：mask_dim = -1, 表示在最后一维上做 mask 操作.

  Returns:

  """
  if not mask_dim < 0:
    raise Exception(f"Mask dim only supports negative numbers! Mask dim: {mask_dim} ")

  for i in range(-mask_dim - 1):
    tensor_mask = tensor_mask.unsqueeze(-1)
  return tensor * tensor_mask



################################################################################
############################# model 优化 ########################################
################################################################################






################################################################################
############################# 简单功能 ########################################
################################################################################

def multiply(*args):
  """
  连乘。
  Args:
    *args:

  Returns:

  """
  result = 1
  for v in args:
    result *= v
  return result