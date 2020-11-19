# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import os
import tqdm
import torch

from lib.data_prerocess.utils import *
from lib.tokenization.bert_finetune_tokenization import *
from my_py_toolkit.file.file_toolkit import make_path_legal
from torch.utils.data import TensorDataset, DataLoader

def read_train_data(input_file):
  with open(input_file, 'r', encoding="utf-8") as f:
    train_data = json.load(f)
    return train_data['data']


def json2features(input_file, output_files, tokenizer, is_training=False,
                  repeat_limit=3, max_query_length=64,
                  max_seq_length=512, doc_stride=128,
                  save_feature=True):

  train_data = read_train_data(input_file)

  examples = generate_examples(is_training, output_files, repeat_limit,
                               train_data)

  # to features
  features = []
  unique_id = 1000000000
  for (example_index, example) in enumerate(tqdm(examples)):
    query_tokens = tokenizer.tokenize(example['question'])
    # query_ids, query_segment_ids, query_mask = gen_bert_input(max_query_length, tokenizer, query_tokens)
    all_doc_tokens, orig_to_tok_index, tok_to_orig_index = tokenize(example['doc_tokens'], tokenizer)

    tok_end_position, tok_start_position = get_answer_scope(all_doc_tokens, example, is_training, orig_to_tok_index,
                                                            tokenizer)

    doc_spans = scope_truncate_context(all_doc_tokens, doc_stride, max_seq_length - len(query_tokens) - 3)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      feature = gen_feature(all_doc_tokens, doc_span, doc_span_index, doc_spans,
                            example_index, is_training, max_seq_length, query_tokens,
                            tok_end_position, tok_start_position, tok_to_orig_index,
                            tokenizer, unique_id)
      features.append(feature)
      unique_id += 1

  print('features num:', len(features))
  if save_feature:
    json.dump(features, open(output_files[1], 'w'))
  return features




def load_data(config, mode="train"):
  tokenizer = FullTokenizer(
    vocab_file=config.bert_config.vocab_file, do_lower_case=config.do_lower_case)
  cur_cfg = config.data_cfg.get(mode, {})
  feature_path = cur_cfg.get("feature_path")
  data_file = cur_cfg.get("data_file")
  is_train = cur_cfg.get("is_train")

  make_path_legal(feature_path)
  if not os.path.exists(feature_path) or config.re_gen_feature:
    train_features = json2features(data_file,
                  [feature_path.replace('_features_', '_examples_'),
                   feature_path],
                  tokenizer, is_training=is_train,
                  max_seq_length=config.bert_config.max_position_embeddings,
                  max_query_length=config.max_query_length,
                  save_feature=config.save_feature)
  # if not os.path.exists(config.dev_dir1) or not os.path.exists(config.dev_dir2):
  #   json2features(config.dev_file, [config.dev_dir1, config.dev_dir2],
  #                 tokenizer,
  #                 is_training=False,
  #                 max_seq_length=config.bert_config.max_position_embeddings)
  else:
    train_features = json.load(open(feature_path, 'r'))
  # dev_examples = json.load(open(config.dev_dir1, 'r'))
  # dev_features = json.load(open(config.dev_dir2, 'r'))
  all_input_ids = torch.tensor([f['input_ids'] for f in train_features],
                               dtype=torch.long)
  all_input_mask = torch.tensor([f['input_mask'] for f in train_features],
                                dtype=torch.long)
  all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features],
                                 dtype=torch.long)

  answer = torch.tensor([f['answer'] for f in train_features],
                               dtype=torch.long)
  # true label
  all_start_positions = torch.tensor(
    [f['start_position'] for f in train_features], dtype=torch.long)
  all_end_positions = torch.tensor([f['end_position'] for f in train_features],
                                   dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_start_positions, all_end_positions, answer)
  train_dataloader = DataLoader(train_data, batch_size=config.batch_size,
                                shuffle=True)
  return tokenizer, train_dataloader

