# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import torch
import torch.nn.functional as F
import traceback

from lib.config import Config
from lib.data_process import load_data
from lib.model.model_baseline import ModelBaseLine
from lib.train.train_qa import train_qa, eval_qa
from my_py_toolkit.torch.utils import load_model, get_adam_optimizer
from tqdm import tqdm

def main(config):
  model = load_model(ModelBaseLine, config)
  optimizer = get_adam_optimizer(model, config)
  _, dev_data = load_data(config, "dev")
  for epoch in tqdm(range(config.start_epoch, config.epochs),
                    total=config.epochs,
                    initial=config.start_epoch,
                    desc="train"):
    train_qa(model, optimizer, config, epoch)
    eval_qa(model, optimizer, config, epoch, "test")
    eval_qa(model, optimizer, config, epoch, "dev")

if __name__ == "__main__":
  config = Config()
  main(config)