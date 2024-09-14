import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from model import TDformer, ns_TDformer, DCTformer, DishTS_Non, Dis_non
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import pandas as panda

warnings.filterwarnings('ignore')

class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DCTformer': DCTformer,
            'DishTS_Non': DishTS_Non
        }
        predict_model = model_dict[self.args.model].Model(self.args).float()
        norm_model = model_dict[self.args.norm].Model(self.args).float()
        model = Model(self.args, predict_model, norm_model).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


