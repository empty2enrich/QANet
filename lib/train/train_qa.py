# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import json
import numpy as np
import os
import torch
import torch.nn.functional as F
import traceback

from collections import Counter
from lib.config import Config
from lib.data_process import load_data
from my_py_toolkit.torch.utils import save_model
from my_py_toolkit.data_visulization.tensorboard import visual_data

def write2file(content, file_path, is_continue=True):
  """"""
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
  mode = "a" if is_continue else "w"
  with open(file_path, mode, encoding="utf-8")as f:
    f.write("\n" + content)

def record_info(losses=[], f1=[], em=[], valid_result={}, epoch=0,
                r_type="train", is_continue=True):
  """
  记录训练中的 loss, f1, em 值.
  Args:
    losses:
    f1:
    em:
    r_type:

  Returns:

  """
  dir_name = f"./log/{r_type}/"
  losses = [str(v) for v in losses]
  f1 = [str(v) for v in f1]
  em = [str(v) for v in em]
  if losses:
    write2file(",".join(losses), f"{dir_name}losses.txt", is_continue=is_continue)
  if f1:
    write2file(",".join(f1), f"{dir_name}f1.txt", is_continue=is_continue)
  if em:
    write2file(",".join(em), f"{dir_name}em.txt", is_continue=is_continue)

  if valid_result:
    write2file(json.dumps(valid_result, ensure_ascii=False, indent=2),
               f"{dir_name}valid_result_{epoch}.json", is_continue=is_continue)


def find_max_porper(start_index_softmax, end_index_softmax):
  """
  根据 start index 与 end index 的 softmax 找出最大的联合概率: max(start_index_pro * end_index_pro)
  Args:
    start_index_softmax(dim):
    end_index_softmax(dim):

  Returns:

  """
  # b_max_pro_index[index] mean: b_softmax[index:] 中可能性最大位置的下标.
  b_max_pro_index = [-1] * end_index_softmax.shape[0]
  b_max_pro = -1
  b_max_index = -1
  for index in range(end_index_softmax.shape[0] - 1, -1, -1): # 0.1 左右
    if end_index_softmax[index] > b_max_pro:
      b_max_pro = end_index_softmax[index]
      b_max_index = index
    b_max_pro_index[index] = b_max_index


  max_start_index = -1
  max_end_index = -1
  max_pro = -1
  for start_index in range(start_index_softmax.shape[0]):
    max_start_pro = start_index_softmax[start_index]
    max_pro_end_index = b_max_pro_index[start_index]
    max_end_pro = end_index_softmax[max_pro_end_index]
    #  * 和 torch.mul 执行时间差不多
    cur_max_pro = torch.mul(max_start_pro, max_end_pro)
    if cur_max_pro > max_pro:
      max_pro = cur_max_pro
      max_start_index = start_index
      max_end_index = max_pro_end_index

  return max_start_index, max_end_index, max_pro


def find_max_proper_batch(start_softmax, end_softmax):
  """

  Args:
    start_softmax(batch_size, dim):
    end_softmax(batch_size, dim):

  Returns:

  """
  start_index = []
  end_index = []
  max_pro = []
  for index in range(start_softmax.shape[0]):
    start, end, pro = find_max_porper(start_softmax[index], end_softmax[index])
    start_index.append(start)
    end_index.append(end)
    max_pro.append(pro)

  return start_index, end_index, max_pro


def convert_pre_res(input_ids, pre_start, pre_end, ori_start, ori_end, probabilities, tokenizer):
  """"""
  result = []
  for input, p_start, p_end, o_start, o_end, probability in zip(input_ids, pre_start, pre_end, ori_start, ori_end, probabilities):
    o_start, o_end, probability = o_start.tolist(), o_end.tolist(), probability.tolist()
    tokens = tokenizer.convert_ids_to_tokens(input.tolist())
    question = "".join(tokens[:tokens.index("[SEP]")])
    context = "".join(tokens[tokens.index("[SEP]"):])
    label_answer = "".join(
      tokens[o_start:o_end + 1])
    predict_answer = "".join(tokens[p_start:p_end + 1])
    cur_res = {
      "context": context,
      "question": question,
      "label_answer": label_answer,
      "predict_answer": predict_answer,
      "is_correct": label_answer == predict_answer,
      "probability": float(probability)
    }
    result.append(cur_res)

  return result

def exact_match_score(prediction, ground_truth):
  # return (normalize_answer(prediction) == normalize_answer(ground_truth))
  return prediction == ground_truth


def f1_score(prediction, ground_truth):
  """

  Args:
    prediction(str):
    ground_truth(str):

  Returns:

  """
  # prediction_tokens = normalize_answer(prediction).split()
  # ground_truth_tokens = normalize_answer(ground_truth).split()
  prediction_tokens = list(prediction)
  ground_truth_tokens = list(ground_truth)
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  """
  计算准确率.
  Args:
    metric_fn:
    prediction(str):
    ground_truths(str):

  Returns:

  """
  return metric_fn(prediction, ground_truths)

def evaluate_valid_result(valid_result, exact_match_total=0, f1_total=0, total=0):
  # f1 = exact_match = total = 0
  for item in valid_result:
    total += 1
    ground_truths = item.get("label_answer")
    prediction = item.get("predict_answer")
    exact_match_total += metric_max_over_ground_truths(exact_match_score, prediction,
                                                       ground_truths)
    f1_total += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
  exact_match = 100.0 * exact_match_total / total
  f1 = 100.0 * f1_total / total
  return exact_match_total, f1_total, exact_match, f1



def train_qa(model, optimizer, config, epoch, scheduler=None):
  """
  train model
  Args:
    model:
    config(Config):

  Returns:

  """
  model.train()
  losses = []
  exact_match_total = 0
  f1_total = 0
  softmax = torch.nn.Softmax(dim=-1)
  log_sofmax = torch.nn.LogSoftmax(dim=-1)
  tokenizer, train_data = load_data(config, "train")
  config.logger.info("Start train =============================")
  for step, batch in enumerate(train_data):
    try:
      optimizer.zero_grad()
      batch = tuple([v.to(config.device) for v in batch])
      input_ids, input_mask, segment_ids, start_positions, end_positions = batch
      input_mask = input_mask.float()
      start_embeddings, end_embeddings = model(input_ids, input_mask, segment_ids)
      loss = calculate_loss(end_embeddings, end_positions, log_sofmax,
                            start_embeddings, start_positions)
      config.logger.info(f"Loss: {loss}, epoch:{epoch}, step: {step}")
      losses.append(loss.item())

      exact_match_total, f1_total = record_train_info_calculate_f1_em(config, end_embeddings, end_positions, epoch,
                                                                      exact_match_total, f1_total, input_ids, loss, model,
                                                                      optimizer, softmax, start_embeddings, start_positions,
                                                                      step, tokenizer, mode="train")
      # 参数更新
      loss.backward()
      optimizer.step()
      if scheduler:
        scheduler.step()
      if step % config.record_interval_steps == 0:
        save_model(model, config.model_save_dir, optimizer, epoch=epoch, steps=step)
      # record
    except Exception:
      config.logger.error(traceback.format_exc())

  loss_avg = np.mean(losses)
  config.logger.info("Epoch {:8d} loss {:8f}\n".format(epoch, loss_avg))

def eval_qa(model, optimizer, config, epoch, mode="test"):
  """
  train model
  Args:
    model:
    config(Config):

  Returns:

  """
  model.eval()
  losses = []
  exact_match_total = 0
  f1_total = 0
  softmax = torch.nn.Softmax(dim=-1)
  log_sofmax = torch.nn.LogSoftmax(dim=-1)
  tokenizer, train_data = load_data(config, mode)
  config.logger.info("Start train =============================")
  for step, batch in enumerate(train_data):
    try:
      batch = tuple([v.to(config.device) for v in batch])
      input_ids, input_mask, segment_ids, start_positions, end_positions = batch
      input_mask = input_mask.float()
      start_embeddings, end_embeddings = model(input_ids, input_mask, segment_ids)
      loss = calculate_loss(end_embeddings, end_positions, log_sofmax,
                            start_embeddings, start_positions)
      config.logger.info(f"Loss: {loss}, epoch:{epoch}, step: {step}")
      losses.append(loss.item())
      #
      exact_match_total, f1_total = record_train_info_calculate_f1_em(config, end_embeddings, end_positions, epoch,
                                                                      exact_match_total, f1_total, input_ids, loss, model,
                                                                      optimizer, softmax, start_embeddings, start_positions,
                                                                      step, tokenizer, mode)
      # record
    except Exception:
      config.logger.error(traceback.format_exc())

  loss_avg = np.mean(losses)
  config.logger.info("Epoch {:8d} loss {:8f}\n".format(epoch, loss_avg))


def record_train_info_calculate_f1_em(config, end_embeddings, end_positions, epoch,
                                      exact_match_total, f1_total, input_ids, loss, model,
                                      optimizer, softmax, start_embeddings, start_positions,
                                      step, tokenizer, mode="train"):
  pre_start, pre_end, probabilities = find_max_proper_batch(
    softmax(start_embeddings), softmax(end_embeddings))
  cur_res = convert_pre_res(input_ids, pre_start, pre_end, start_positions,
                            end_positions, probabilities, tokenizer)
  exact_match_total, f1_total, exact_match, f1 = evaluate_valid_result(
    cur_res, exact_match_total, f1_total, (step + 1) * config.batch_size)
  record_info(valid_result=cur_res, epoch=epoch, is_continue=True, r_type=mode)
  if mode == "train":
    visual_data(model, epoch, step, loss, optimizer,
                exact_match_total, f1_total, exact_match, f1, model,
                config.visual_gradient, config.visual_gradient_dir,
                  config.visual_parameter, config.visual_parameter_dir,
                  config.visual_loss, config.visual_loss_dir,
                  config.visual_optimizer, config.visual_optimizer_dir,
                  config.visual_valid_result, config.visual_valid_result_dir)
  else:
    visual_data(model, epoch, step, loss, optimizer,
              exact_match_total, f1_total, exact_match, f1, model)
  return exact_match_total, f1_total


def calculate_loss(end_embeddings, end_positions, log_sofmax, start_embeddings,
                   start_positions):
  loss_start = F.nll_loss(log_sofmax(start_embeddings), start_positions,
                          reduction="mean")
  loss_end = F.nll_loss(log_sofmax(end_embeddings), end_positions,
                        reduction="mean")
  loss = (loss_start + loss_end) / 2
  return loss