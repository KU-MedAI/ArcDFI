import torch
import torch.nn as nn
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

import numpy as np
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score as aup
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index as c_index
import pandas as pd
from datetime import datetime

from time import sleep
from typing import Any, Callable, List, Tuple, Union


class Masked_NLLLoss(nn.Module):
    def __init__(self, weight=[], device='cuda'):
        super(Masked_NLLLoss, self).__init__()
        if len(weight) == 0:
            self.criterion = nn.NLLLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(reduction='none', weight=torch.FloatTensor(weight).to(device))

    def forward(self, pred, label):
        pred_  = (pred.view(pred.size(0)*pred.size(1),pred.size(2))).log()
        label_ = label.reshape(-1)
        label_ = (label_ + 1.)/2
        loss   = self.criterion(pred_, label_)[label_!=0.5].mean()

        return loss  

class DummyScheduler:
    def __init__(self):
        x = 0
    def step(self):
        return 

def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return [element.detach().cpu().numpy() for element in tensor]
    else:
        return tensor

def initialize(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)