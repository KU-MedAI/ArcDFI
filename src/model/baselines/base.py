import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import *

import lightning as L
from torchmetrics.classification import *

FEAT2DIM   = dict(morgan=1024,pharma=39972,maccs=167,erg=441,pubchem=881)

def load_to_device(batch_dict, device):
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device)

    return batch_dict


class CASTERAuxiliaryLoss(nn.Module):
    def __init__(
        self, recon_loss_coeff: float = 1e-1, proj_coeff: float = 1e-1, lambda1: float = 1e-2, lambda2: float = 1e-1
    ):
        """
        Initialize the custom loss function for the supervised learning stage of the CASTER algorithm.

        :param recon_loss_coeff: coefficient for the reconstruction loss
        :param proj_coeff: coefficient for the projection loss
        :param lambda1: regularization coefficient for the projection loss
        :param lambda2: regularization coefficient for the augmented projection loss
        """
        super().__init__()
        self.recon_loss_coeff = recon_loss_coeff
        self.proj_coeff = proj_coeff
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss = torch.nn.BCELoss()

    def forward(self, recon, code, dictionary_features_latent, drug_pair_features_latent, drug_pair_features):
        """Perform a forward pass of the loss calculation for the supervised learning stage of the CASTER algorithm.

        :param x: a tuple of tensors returned by the model forward pass (see CASTER.forward() method)
        :param target: target labels
        :return: combined loss value
        """
        batch_size, _ = drug_pair_features.shape
        # loss_prediction = self.loss(score, target.float())
        loss_reconstruction = self.recon_loss_coeff * self.loss(recon, drug_pair_features)
        loss_projection = self.proj_coeff * (
            torch.norm(drug_pair_features_latent - torch.matmul(code, dictionary_features_latent))
            + self.lambda1 * torch.sum(torch.abs(code)) / batch_size
            + self.lambda2 * torch.norm(dictionary_features_latent, p="fro") / batch_size
        )
        loss = loss_reconstruction + loss_projection

        return loss

class LossModule(nn.Module):
    def __init__(self, conf):
        super(LossModule, self).__init__()

        self.loss_criterion            = nn.ModuleDict()
        self.loss_criterion['dfi']     = nn.BCELoss()

    def forward(self, **kwargs):
        loss_dict = dict()
        loss_dict['dfi']     = self.loss_criterion['dfi'](kwargs['yhat_dfi'], kwargs['y_dfi'])

        if 'caster_reconstructed' in kwargs.keys():
            aux_loss = CASTERAuxiliaryLoss()
            loss_dict['dfi'] = loss_dict['dfi'] + aux_loss(kwargs['caster_reconstructed'],
                                                           kwargs['caster_code'],
                                                           kwargs['caster_dictionary_features_latent'],
                                                           kwargs['caster_drug_pair_features_latent'],
                                                           kwargs['caster_drug_pair_features'])

        return loss_dict


class BaseLightningModel(L.LightningModule):
    def __init__(self, conf):
        super(BaseLightningModel, self).__init__()

        self.train_params = conf.train_params
        self.loss_module  = LossModule(conf)

        self.metric_module = nn.ModuleDict()
        for m in ['train', 'valid', 'test']:
            for n in ['dfi']:
                self.metric_module[f'{m}/{n}/accuracy']  = BinaryAccuracy()
                self.metric_module[f'{m}/{n}/auroc']     = BinaryAUROC()
                self.metric_module[f'{m}/{n}/f1']        = BinaryF1Score()
                self.metric_module[f'{m}/{n}/precision'] = BinaryPrecision()
                self.metric_module[f'{m}/{n}/recall']    = BinaryRecall()
                self.metric_module[f'{m}/{n}/auprc']     = BinaryAveragePrecision()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.train_params.learning_rate, 
                                      weight_decay=self.train_params.weight_decay)

        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        output  = self.forward(batch)

        losses  = self.loss_module(**output)
        loss    = self.train_params.loss_coef.dfi*losses['dfi'] 
        for k,v in losses.items(): self.log(f'train/loss/{k}', v.item())
        self.log('train/loss/all', loss.item(), sync_dist=True)
        
        self.metric_module['train/dfi/accuracy'](output['yhat_dfi'], output['y_dfi'])
        self.log('train/dfi/accuracy', self.metric_module['train/dfi/accuracy'])
        for m in ['auroc', 'f1', 'precision', 'recall']:
            self.metric_module[f'train/dfi/{m}'].update(output['yhat_dfi'], output['y_dfi'])
        self.metric_module['train/dfi/auprc'].update(output['yhat_dfi'], output['y_dfi'].long())

        return loss

    def on_train_epoch_end(self):
        for m in ['dfi']:
            self.metric_module[f'train/{m}/accuracy'].reset()
            for n in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.log(f'train/{m}/{n}', self.metric_module[f'train/{m}/{n}'].compute(), sync_dist=True)


    def validation_step(self, batch, batch_idx):
        output  = self.forward(batch)

        losses  = self.loss_module(**output)
        loss    = self.train_params.loss_coef.dfi*losses['dfi']
        for k,v in losses.items(): self.log(f'valid/loss/{k}', v.item())
        self.log('valid/loss/all', loss.item(), sync_dist=True)
        
        self.metric_module['valid/dfi/accuracy'](output['yhat_dfi'], output['y_dfi'])
        self.log('valid/dfi/accuracy', self.metric_module['valid/dfi/accuracy'])
        for m in ['auroc', 'f1', 'precision', 'recall']:
            self.metric_module[f'valid/dfi/{m}'].update(output['yhat_dfi'], output['y_dfi'])
        self.metric_module['valid/dfi/auprc'].update(output['yhat_dfi'], output['y_dfi'].long())

        return loss


    def on_validation_epoch_end(self):
        for m in ['dfi']:
            self.metric_module[f'valid/{m}/accuracy'].reset()
            for n in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.log(f'valid/{m}/{n}', self.metric_module[f'valid/{m}/{n}'].compute(), sync_dist=True)


    def test_step(self, batch, batch_idx):
        output  = self.forward(batch)

        losses  = self.loss_module(**output)
        loss    = self.train_params.loss_coef.dfi*losses['dfi'] 
        for k,v in losses.items(): self.log(f'test/loss/{k}', v.item())
        self.log('test/loss/all', loss.item(), sync_dist=True)
        
        self.metric_module['test/dfi/accuracy'](output['yhat_dfi'], output['y_dfi'])
        self.log('test/dfi/accuracy', self.metric_module['test/dfi/accuracy'])
        for m in ['auroc', 'f1', 'precision', 'recall']:
            self.metric_module[f'test/dfi/{m}'].update(output['yhat_dfi'], output['y_dfi'])
        self.metric_module['test/dfi/auprc'].update(output['yhat_dfi'], output['y_dfi'].long())

        return loss


    def on_test_epoch_end(self):
        for m in ['dfi']:
            self.metric_module[f'test/{m}/accuracy'].reset()
            for n in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.log(f'test/{m}/{n}', self.metric_module[f'test/{m}/{n}'].compute(), sync_dist=True)


    def forward(self, input_batch):

        raise 




# def register_hooks(model):
#     """
#     Register hooks to check for unused parameters in the model.
#     """
#     def hook(module, input, output):
#         module._used = True

#     for name, param in model.named_parameters():
#         param._used = False  # Mark all parameters as unused initially

#     for name, module in model.named_modules():
#         if len(list(module.parameters())) > 0:
#             module.register_forward_hook(hook)

# def check_unused_parameters(model):
#     """
#     Print parameters that are unused after a forward and backward pass.
#     """
#     for name, param in model.named_parameters():
#         if not param._used:
#             print(f"Parameter {name} is unused.")








"""Base classes for models and utilities."""

from abc import ABC, abstractmethod

# from chemicalx.data import DrugPairBatch

import torch
import torch.nn as nn
import torch.optim as optim
import pickle


class UnimplementedModel:
    """The base class for unimplemented ChemicalX models."""

    def __init__(self, x: int):
        """Instantiate a base model."""
        self.x = x


class Model(nn.Module, ABC):
    """The base class for ChemicalX models."""