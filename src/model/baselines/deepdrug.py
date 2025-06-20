import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_max_pool
from torch_geometric.utils import from_smiles
from torch_geometric.data.collate import collate
# from torchdrug.layers import GraphConv, MaxReadout
from .base import *


class DeepDrug(BaseLightningModel):
    """An implementation of the DeepDrug model from [cao2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/14

    .. [cao2020] Cao, X., *et al.* (2020). `DeepDrug: A general graph-based deep learning framework
       for drug relation prediction <https://doi.org/10.1101/2020.11.09.375626>`_.
       *bioRxiv*, 2020.11.09.375626.
    """


    def __init__(
        self,
        conf,
        num_gcn_layers: int = 4,
        gcn_layer_hidden_size: int = 64,
        out_channels: int = 1,
        dropout_rate: float = 0.1,
    ):
        """Instantiate the DeepDrug model.

        :param molecule_channels: The number of molecular features.
        :param num_gcn_layers: Number of GCN layers.
        :param gcn_layer_hidden_size: number of hidden units in GCN layers
        :param out_channels: The number of output channels.
        :param dropout_rate: Dropout rate on the final fully-connected layer.
        """
        super(DeepDrug, self).__init__(conf)
        ############################################################# 
        self.module    = nn.ModuleDict()
        self.feat_type = conf.model_params.compound_features
        self.cyp_dim   = 10 if conf.model_params.use_cyp_label else 0 
        molecule_channels  = FEAT2DIM[self.feat_type] + self.cyp_dim
        self.module['embed'] = torch.nn.ModuleList([])
        embedding_dimensions = [121, 10, 12, 13, 10, 6, 9, 3, 3]
        for i, d in enumerate(embedding_dimensions):
            self.module['embed'].append(nn.Embedding(d, gcn_layer_hidden_size*2))
        #############################################################

        self.num_gcn_layers = num_gcn_layers
        self.gcn_layer_hidden_size = gcn_layer_hidden_size
        self.module['first'] = GCNConv(self.gcn_layer_hidden_size*2, self.gcn_layer_hidden_size)

        # add remaining GCN layers
        self.module['layers'] = torch.nn.ModuleList([])
        for _ in range(num_gcn_layers - 1):
            self.module['layers'].append(GCNConv(self.gcn_layer_hidden_size, self.gcn_layer_hidden_size))
            self.module['layers'].append(nn.BatchNorm1d(self.gcn_layer_hidden_size))

        self.readout = global_max_pool
        self.middle_channels = 2 * self.gcn_layer_hidden_size + self.cyp_dim + self.cyp_dim

        self.module['final'] = nn.Sequential(
            nn.BatchNorm1d(self.middle_channels),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.middle_channels, out_channels),
            nn.Sigmoid(),
        )

    def _forward_molecules(self, x, edge_index, batch):
        features = self.module['first'](x, edge_index)
        for layer in self.module['layers']:
            if isinstance(layer, nn.BatchNorm1d):
                features = layer(features)
            else:
                features = layer(features, edge_index)

        return self.readout(features, batch)

    def _combine_sides(self, left, right):
        return torch.cat([left, right], dim=1)

    def forward(self, input_batch):
        input_batch     = load_to_device(input_batch, 'cuda')
        """
        Run a forward pass of the DeepDrug model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.

        :return: A column vector of predicted synergy scores.
        """
        # Prepare Input Tensors
        list_dcomp_data = [from_smiles(x) for x in input_batch['dcomp_smiles']]
        dcomp_batch = collate(list_dcomp_data[0].__class__, list_dcomp_data, False, True)[0].to(self.device)
        list_fcomp_data = [from_smiles(x) for x in input_batch['fcomp_smiles']]
        fcomp_batch = collate(list_dcomp_data[0].__class__, list_fcomp_data, False, True)[0].to(self.device)


        # Main Model Forward Part
        dcomp_features = []
        for i in range(9):
            dcomp_features.append(self.module['embed'][i](dcomp_batch.x[:,i]).unsqueeze(2))
        dcomp_features = torch.cat(dcomp_features, dim=2).mean(2)
        fcomp_features = []
        for i in range(9):
            fcomp_features.append(self.module['embed'][i](fcomp_batch.x[:,i]).unsqueeze(2))
        fcomp_features = torch.cat(fcomp_features, dim=2).mean(2)

        dcomp_latent = self._forward_molecules(dcomp_features, dcomp_batch.edge_index, dcomp_batch.batch)
        fcomp_latent = self._forward_molecules(fcomp_features, fcomp_batch.edge_index, fcomp_batch.batch)
        hidden = self._combine_sides(dcomp_latent, fcomp_latent)
        if self.cyp_dim > 0:
            hidden = torch.cat([hidden, 
                                input_batch[f'dcomp_dci_labels'],
                                input_batch[f'dcomp_dci_labels']],dim=1)

        dfi_predicted = self.module['final'](hidden)

        # Wrap Up the Results
        output_batch  = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                             fcomp_smiles=input_batch['fcomp_smiles'],
                             y_dfi=input_batch['y_dfi_label'],
                             yhat_dfi=dfi_predicted.reshape(-1))

        return output_batch