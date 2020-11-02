# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import os
import tqdm

from lib.tokenization.bert_finetune_tokenization import *


################################################################################
######################## 数据预处理公共方法 ##############################
################################################################################



SPIECE_UNDERLINE = '▁'

def is_chinese_char(cp):
  if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
    (cp >= 0x3400 and cp <= 0x4DBF) or  #
    (cp >= 0x20000 and cp <= 0x2A6DF) or  #
    (cp >= 0x2A700 and cp <= 0x2B73F) or  #
    (cp >= 0x2B740 and cp <= 0x2B81F) or  #
    (cp >= 0x2B820 and cp <= 0x2CEAF) or
    (cp >= 0xF900 and cp <= 0xFAFF) or  #
    (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
    return True

  return False

def is_fuhao(c):
  if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
    or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
    or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
    or c == '‘' or c == '’':
    return True
  return False

def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
    c) == 0x202F or c == SPIECE_UNDERLINE:
    return True
  return False

def tokenize_chinese_chars(text):
  """Adds whitespace around any CJK character."""
  output = []
  for char in text:
    cp = ord(char)
    if is_chinese_char(cp) or is_fuhao(char):
      if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
        output.append(SPIECE_UNDERLINE)
      output.append(char)
      output.append(SPIECE_UNDERLINE)
    else:
      output.append(char)
  return "".join(output)


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def tokenize_context(context):
  """
  生产 context 的 token。
  Args:
    context:

  Returns:

  """
  context_chs = tokenize_chinese_chars(context)
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in context_chs:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    if c != SPIECE_UNDERLINE:
      char_to_word_offset.append(len(doc_tokens) - 1)
  return char_to_word_offset, doc_tokens



def generate_example(char_to_word_offset, context, doc_tokens,
                     is_training, qas, repeat_limit):
  """
  根据 question、answer 生产 example。
  Args:
    char_to_word_offset: char 在 doc token 的 index.
    context:
    doc_tokens:
    is_training:
    qas:
    repeat_limit:

  Returns:

  """
  qid = qas['id']
  ques_text = qas['question']
  ans_text = qas['answers'][0]['text']
  start_position_final = None
  end_position_final = None
  mis_match = False
  if is_training:
    count_i = 0
    start_position = qas['answers'][0]['answer_start']

    end_position = start_position + len(ans_text) - 1
    while context[
          start_position:end_position + 1] != ans_text and count_i < repeat_limit:
      start_position -= 1
      end_position -= 1
      count_i += 1

    while context[start_position] == " " or context[
      start_position] == "\t" or \
      context[start_position] == "\r" or context[
      start_position] == "\n":
      start_position += 1

    start_position_final = char_to_word_offset[start_position]
    end_position_final = char_to_word_offset[end_position]

    if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".",
                                            ","}:
      start_position_final += 1

    actual_text = "".join(
      doc_tokens[start_position_final:(end_position_final + 1)])
    cleaned_answer_text = "".join(
      whitespace_tokenize(ans_text))

    if actual_text != cleaned_answer_text:
      print(actual_text, 'V.S', cleaned_answer_text)
      mis_match = True
      # ipdb.set_trace()
  example = {'doc_tokens': doc_tokens,
                   'orig_answer_text': ans_text,
                   'qid': qid,
                   'question': ques_text,
                   'answer': ans_text,
                   'start_position': start_position_final,
                   'end_position': end_position_final}
  return example, mis_match


def generate_examples(is_training, output_files, repeat_limit, train_data):
  """
  生成 examples.
  Args:
    is_training:
    output_files:
    repeat_limit:
    train_data:

  Returns:

  """
  # to examples
  examples = []
  mis_match = 0
  for article in tqdm(train_data):
    for para in article['paragraphs']:
      context = para['context']

      char_to_word_offset, doc_tokens = tokenize_context(context)

      for qas in para['qas']:
        example, is_mis_match = generate_example(char_to_word_offset, context,
                                                 doc_tokens,
                                                 is_training, qas, repeat_limit)
        examples.append(example)
  print('examples num:', len(examples))
  print('mis_match:', mis_match)
  os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
  json.dump(examples, open(output_files[0], 'w'))
  return examples

def merge_tokens(*tokens):
  tokens_merged = []
  segment_ids = []
  token_to_orig_map_merged = []
  token_is_max_context = []