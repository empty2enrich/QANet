# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import os
import torch
from lib.tokenization.bert_finetune_tokenization import *
from torch.utils.data import TensorDataset, DataLoader

################################################################################
######################## bert finetune preprocess ##############################
################################################################################

def load_data(config):
  tokenizer = FullTokenizer(
    vocab_file=config.bert_config.vocab_file, do_lower_case=config.do_lower_case)
  if not os.path.exists(config.train_dir):
    json2features(config.train_file,
                  [config.train_dir.replace('_features_', '_examples_'),
                   config.train_dir],
                  tokenizer, is_training=True,
                  max_seq_length=config.bert_config.max_position_embeddings)
  if not os.path.exists(config.dev_dir1) or not os.path.exists(config.dev_dir2):
    json2features(config.dev_file, [config.dev_dir1, config.dev_dir2],
                  tokenizer,
                  is_training=False,
                  max_seq_length=config.bert_config.max_position_embeddings)
  train_features = json.load(open(feature_path, 'r'))
  # dev_examples = json.load(open(config.dev_dir1, 'r'))
  # dev_features = json.load(open(config.dev_dir2, 'r'))
  all_input_ids = torch.tensor([f['input_ids'] for f in train_features],
                               dtype=torch.long)
  all_input_mask = torch.tensor([f['input_mask'] for f in train_features],
                                dtype=torch.long)
  all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features],
                                 dtype=torch.long)
  # true label
  all_start_positions = torch.tensor(
    [f['start_position'] for f in train_features], dtype=torch.long)
  all_end_positions = torch.tensor([f['end_position'] for f in train_features],
                                   dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_start_positions, all_end_positions)
  train_dataloader = DataLoader(train_data, batch_size=config.n_batch,
                                shuffle=True)
  ############################################################################
  ############################################################################
  ############################################################################
  return tokenizer, train_dataloader
