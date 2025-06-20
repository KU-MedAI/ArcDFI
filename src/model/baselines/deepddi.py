from .base import *


class DeepDDI(BaseLightningModel):
    """An implementation of the DeepDDI model from [ryu2018]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/2

    .. [ryu2018] Ryu, J. Y., *et al.* (2018). `Deep learning improves prediction
       of drug–drug and drug–food interactions <https://doi.org/10.1073/pnas.1803294115>`_.
       *Proceedings of the National Academy of Sciences*, 115(18), E4304–E4311.
    """

    def __init__(
        self,
        conf,
        hidden_channels: int = 2048,
        hidden_layers_num: int = 9,
        out_channels: int = 1,
    ):
        """Instantiate the DeepDDI model.

        :param drug_channels: The number of drug features.
        :param hidden_channels: The number of hidden layer neurons.
        :param hidden_layers_num: The number of hidden layers.
        :param out_channels: The number of output channels.
        """
        super().__init__(conf)
        ############################################################# 
        self.module    = nn.ModuleDict()
        self.feat_type = conf.model_params.compound_features
        self.cyp_dim   = 10 if conf.model_params.use_cyp_label else 0 
        drug_channels  = FEAT2DIM[self.feat_type] + self.cyp_dim
        #############################################################

        assert hidden_layers_num > 1
        layers = [
            nn.Linear(drug_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers_num - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=hidden_channels, affine=True, momentum=None),
                    nn.ReLU(),
                ]
            )
        layers.extend([nn.Linear(hidden_channels, out_channels), nn.Sigmoid()])
        self.module['final'] = nn.Sequential(*layers)

    def _combine_sides(self, left, right):
        return torch.cat([left, right], dim=1)

    def forward(self, input_batch):
        input_batch     = load_to_device(input_batch, 'cuda')
        """Run a forward pass of the DeepDDI model.

        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted interaction scores.
        """
        # Prepare Input Tensors
        # Prepare Input Tensors
        if self.cyp_dim > 0:
            dcomp_input = torch.cat([input_batch[f'dcomp_{self.feat_type}_fp'],
                                     input_batch[f'dcomp_dci_labels']], dim=1)
            fcomp_input = torch.cat([input_batch[f'fcomp_{self.feat_type}_fp'],
                                     input_batch[f'dcomp_dci_labels']], dim=1)
        else:
            dcomp_input = input_batch[f'dcomp_{self.feat_type}_fp']
            fcomp_input = input_batch[f'fcomp_{self.feat_type}_fp']
        hidden = self._combine_sides(dcomp_input, fcomp_input)

        # Main Model Forward Part
        dfi_predicted = self.module['final'](hidden)

        # Wrap Up the Results
        output_batch  = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                             fcomp_smiles=input_batch['fcomp_smiles'],
                             y_dfi=input_batch['y_dfi_label'],
                             yhat_dfi=dfi_predicted.reshape(-1))

        return output_batch