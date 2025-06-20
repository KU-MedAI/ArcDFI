import torch
from torch import nn
from .base import *

class DeepSynergy(BaseLightningModel):
    r"""An implementation of the DeepSynergy model from [preuer2018]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/16

    .. [preuer2018] Preuer, K., *et al.* (2018). `DeepSynergy: predicting anti-cancer drug synergy
       with Deep Learning <https://doi.org/10.1093/bioinformatics/btx806>`_. *Bioinformatics*, 34(9), 1538–1546.
    """

    def __init__(
        self,
        conf,
        input_hidden_channels: int = 32,
        middle_hidden_channels: int = 32,
        final_hidden_channels: int = 32,
        out_channels: int = 1,
        dropout_rate: float = 0.5,
    ):
        """Instantiate the DeepSynergy model.

        :param context_channels: The number of context features.
        :param drug_channels: The number of drug features.
        :param input_hidden_channels: The number of hidden layer neurons in the input layer.
        :param middle_hidden_channels: The number of hidden layer neurons in the middle layer.
        :param final_hidden_channels: The number of hidden layer neurons in the final layer.
        :param out_channels: The number of output channels.
        :param dropout_rate: The rate of dropout before the scoring head is used.
        """
        super().__init__(conf)
        ############################################################# 
        self.module    = nn.ModuleDict()
        self.feat_type = conf.model_params.compound_features
        self.cyp_dim   = 10 if conf.model_params.use_cyp_label else 0 
        drug_channels  = FEAT2DIM[self.feat_type]
        #############################################################

        self.module['final'] = nn.Sequential(
            nn.Linear(drug_channels + drug_channels + self.cyp_dim + self.cyp_dim, input_hidden_channels),
            nn.ReLU(),
            nn.Linear(input_hidden_channels, middle_hidden_channels),
            nn.ReLU(),
            nn.Linear(middle_hidden_channels, final_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden_channels, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, input_batch):
        input_batch    = load_to_device(input_batch, 'cuda')
        """Run a forward pass of the DeepDDI model.

        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted interaction scores.
        """
        
        # Prepare Input Tensors
        if self.cyp_dim > 0:
            dcomp_input = torch.cat([input_batch[f'dcomp_{self.feat_type}_fp'],
                                     input_batch[f'dcomp_dci_labels']], dim=1)
            fcomp_input = torch.cat([input_batch[f'fcomp_{self.feat_type}_fp'],
                                     input_batch[f'dcomp_dci_labels']], dim=1)
        else:
            dcomp_input = input_batch[f'dcomp_{self.feat_type}_fp']
            fcomp_input = input_batch[f'fcomp_{self.feat_type}_fp']
        hidden = torch.cat([dcomp_input, fcomp_input], dim=1)

        # Main Model Forward Part
        dfi_predicted = self.module['final'](hidden)

        # Wrap Up the Results
        output_batch  = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                             fcomp_smiles=input_batch['fcomp_smiles'],
                             y_dfi=input_batch['y_dfi_label'],
                             yhat_dfi=dfi_predicted.reshape(-1))

        return output_batch