# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

from lib.utils import BertConfig
from my_py_toolkit.log.logger import get_logger
from pytorch_transformers.modeling_bert import BertModel

class Config(object):
  def __init__(self):
    self.device = "cuda"
    self.dropout = 0.1

    # data preprocess
    # split_qc 是否把 context token 与 question token 分开
    self.split_qc = True
    # re_gen_feature True: 表示重新生成 feature 缓存文件
    self.re_gen_feature = False
    self.max_query_length = 32
    self.save_feature = True

    # encoder
    self.use_encoder = True
    self.use_position_embedding = True
    self.encoder_hidden_layer_number = 11
    self.encoder_intermediate_dim = 3072
    self.encoder_dropout = 0.1

    # attention
    self.attention_head_num = 16
    self.attention_droup_out = 0.1
    self.attention_use_bias = False
    # qc、cq attention
    self.bi_direction_attention = True
    # cq: context -> question; qc: question -> context
    self.direction = "cq"

    # train
    self.training = True

    # cnn
    self.use_conv=False
    self.chan_in = 64
    self.chan_out = 64
    self.kernel = 3
    self.dim = 2
    self.stride = 2

    # LSTM
    self.use_lstm = False
    self.lstm_hidden_size = 768
    self.lstm_bi_direction = True
    self.lstm_layer_num = 11
    self.lstm_batch_first = False

    # attention match pyramid
    self.pyramid_chan = 64
    self.pyramid_kernel = 3
    self.pyramid_stride = 1
    self.pyramid_dim = 2
    self.pyramid_pool_kernel = 2
    self.pyramid_pool_stride = 2
    self.pyramid_resnet = True

    # bert config
    self.freeze_bert = False
    self.bert_config = BertConfig(bert_path="./resource/bert_model/bert",
                                  bert_class=BertModel,
                                  use_pretrained_bert=True,
                                  freeze_paras=self.freeze_bert)
    self.do_lower_case = True

    # train cfg
    self.learning_rate = 3e-5
    self.batch_size = 2
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.is_only_save_params = True
    self.mode = "train" # train, debug, valid, test
    self.record_interval_steps = 500
    self.model_save_dir = "../model"
    self.is_continue_train = False
    self.continue_checkpoint = 4000
    self.continue_epoch = 0
    self.start_epoch = 0
    self.epochs = 2
    self.log_path = "../log/log.txt"
    self.logger = get_logger(self.log_path)

    # data visualization
    self.visual_data_dir = "./runs"
    self.visual_loss = True
    self.visual_gradient = False
    self.visual_parameter = False
    self.visual_optimizer = False
    self.visual_valid_result = True
    self.visual_gradient_dir = "../log/gradients"
    self.visual_parameter_dir = "../log/parameters"
    self.visual_loss_dir = "../log/losses"
    self.visual_optimizer_dir = "../log/optimizer"
    self.visual_valid_result_dir = "../log/valid"
    ## 可视化卷积
    self.visual_cnn = False
    self.visual_cnn_dir = "../log/cnn"

    # data preprocess
    self.data_cfg = {
      "train": {
        "feature_path": "dataset/cmrc2018/train_features_roberta512.json",
        "data_file": "data/cmrc2018/cmrc2018_train.json",
        "is_train": True
      },
      "test": {
        "feature_path": "dataset/cmrc2018/test_features_roberta512.json",
        "data_file": "data/cmrc2018/cmrc2018_trial.json",
        "is_train": True
      },
      "dev": {
        "feature_path": "dataset/cmrc2018/dev_features_roberta512.json",
        "data_file": "data/cmrc2018/cmrc2018_dev.json",
        "is_train": True
      }
    }


  def load_from_json(self, json_path):
    """"""