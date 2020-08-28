# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

from lib.config import Config
from lib.utils import load_model, get_adam_optimizer
from lib.model.model_baseline import ModelBaseLine
from tqdm import tqdm

def train(model, optimizer, config):
  """
  train model
  Args:
    model:
    config(Config):

  Returns:

  """
  model.train()




def main(config):
  model = load_model(ModelBaseLine, config)
  optimizer = get_adam_optimizer(model, config)
  for epoch in tqdm(range(config.start_epoch, config.epochs),
                    total=config.epochs,
                    initial=config.start_epoch,
                    desc="train"):
    pass

if __name__ == "__main__":
  config = Config()
  main(config)