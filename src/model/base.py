import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import lightning as L
from typing import *

FEAT2DIM   = dict(morgan=4096,pharma=39972,maccs=167,erg=441,pubchem=881)

class DrugFoodInteractionBase(L.LightningModule):
    def __init__(self, conf):
        super(DrugFoodInteractionBase, self).__init__()

        self.train_params = conf.train_params
        self.loss_module  = LossModule(conf)

        self.metric_module = dict()
        for m in ['train', 'valid', 'test']:
            for n in ['dfi', 'dci_sub', 'dci_inh']:
                self.metric_module[f'{m}/{n}/accuracy']  = BinaryAccuracy() 
                self.metric_module[f'{m}/{n}/auroc']     = BinaryAUROC()
                self.metric_module[f'{m}/{n}/f1']        = BinaryF1Score()
                self.metric_module[f'{m}/{n}/precision'] = BinaryPrecision()
                self.metric_module[f'{m}/{n}/recall']    = BinaryRecall()


