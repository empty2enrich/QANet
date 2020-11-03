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
from torch.utils.data import TensorDataset, DataLoader

def read_train_data(input_file):
  with open(input_file, 'r') as f:
    train_data = json.load(f)
    return train_data['data']


def json2features(input_file, output_files, tokenizer, is_training=False,
                  repeat_limit=3, max_query_length=64,
                  max_seq_length=512, doc_stride=128):

  train_data = read_train_data(input_file)

  examples = generate_examples(is_training, output_files, repeat_limit,
                               train_data)

  # to features
  features = []
  unique_id = 1000000000
  for (example_index, example) in enumerate(tqdm(examples)):
    query_tokens = tokenizer.tokenize(example['question'])
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example['doc_tokens']):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training:
      tok_start_position = orig_to_tok_index[
        example['start_position']]  # 原来token到新token的映射，这是新token的起点
      if example['end_position'] < len(example['doc_tokens']) - 1:
        tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = improve_answer_span(
        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
        example['orig_answer_text'])

    # The -3 accounts for [CLS], [SEP] and [SEP]
    # max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # The -2 accounts for [CLS], [SEP]
    max_tokens_for_doc = max_seq_length - 2

    doc_spans = []
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    ######## question tokens
    question_tokens = []
    question_mask = [0 for _ in range(max_query_length)]
    question_segment_ids = []
    tokens.append("[CLS]")
    question_segment_ids.append(0)
    for idx, token in enumerate(query_tokens):
      question_tokens.append(token)
      question_segment_ids.append(0)
      question_mask[idx] = 1
    question_tokens.append("[SEP]")
    question_segment_ids.append(0)


    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      # for token in query_tokens:
      #   tokens.append(token)
      #   segment_ids.append(0)
      # tokens.append("[SEP]")
      # segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
        is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None
      if is_training:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        if tok_start_position == -1 and tok_end_position == -1:
          start_position = 0  # 问题本来没答案，0是[CLS]的位子
          end_position = 0
        else:  # 如果原本是有答案的，那么去除没有答案的feature
          out_of_span = False
          doc_start = doc_span.start  # 映射回原文的起点和终点
          doc_end = doc_span.start + doc_span.length - 1

          if not (
            tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
            out_of_span = True
          if out_of_span:
            start_position = 0
            end_position = 0
          else:
            doc_offset = len(query_tokens) + 2
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

      features.append({'unique_id': unique_id,
                       'example_index': example_index,
                       'doc_span_index': doc_span_index,
                       'tokens': tokens,
                       'question_tokens': question_tokens,
                       'token_to_orig_map': token_to_orig_map,
                       'token_is_max_context': token_is_max_context,
                       'input_ids': input_ids,
                       'input_mask': input_mask,
                       'segment_ids': segment_ids,
                       'question_ids': question_ids,
                       'question_mask': question_mask,
                       'question_segment_ids': question_segment_ids,
                       'start_position': start_position,
                       'end_position': end_position})
      unique_id += 1

  print('features num:', len(features))
  json.dump(features, open(output_files[1], 'w'))



def load_data(config, mode="train"):
  tokenizer = FullTokenizer(
    vocab_file=config.bert_config.vocab_file, do_lower_case=config.do_lower_case)
  cur_cfg = config.data_cfg.get(mode, {})
  feature_path = cur_cfg.get("feature_path")
  data_file = cur_cfg.get("data_file")
  is_train = cur_cfg.get("is_train")
  if not os.path.exists(feature_path):
    json2features(data_file,
                  [feature_path.replace('_features_', '_examples_'),
                   feature_path],
                  tokenizer, is_training=is_train,
                  max_seq_length=config.bert_config.max_position_embeddings)
  # if not os.path.exists(config.dev_dir1) or not os.path.exists(config.dev_dir2):
  #   json2features(config.dev_file, [config.dev_dir1, config.dev_dir2],
  #                 tokenizer,
  #                 is_training=False,
  #                 max_seq_length=config.bert_config.max_position_embeddings)
  train_features = json.load(open(feature_path, 'r'))
  # dev_examples = json.load(open(config.dev_dir1, 'r'))
  # dev_features = json.load(open(config.dev_dir2, 'r'))
  all_input_ids = torch.tensor([f['input_ids'] for f in train_features],
                               dtype=torch.long)
  all_input_mask = torch.tensor([f['input_mask'] for f in train_features],
                                dtype=torch.long)
  all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features],
                                 dtype=torch.long)
  question_input_ids = torch.tensor([f['question_ids'] for f in train_features],
                               dtype=torch.long)
  question_input_mask = torch.tensor([f['question_mask'] for f in train_features],
                                dtype=torch.long)
  question_segment_ids = torch.tensor([f['question_segment_ids'] for f in train_features],
                                 dtype=torch.long)
  # true label
  all_start_positions = torch.tensor(
    [f['start_position'] for f in train_features], dtype=torch.long)
  all_end_positions = torch.tensor([f['end_position'] for f in train_features],
                                   dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             question_input_ids, question_input_mask, question_segment_ids,
                             all_start_positions, all_end_positions)
  train_dataloader = DataLoader(train_data, batch_size=config.batch_size,
                                shuffle=True)
  return tokenizer, train_dataloader

