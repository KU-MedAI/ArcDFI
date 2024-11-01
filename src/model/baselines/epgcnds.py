import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import from_smiles
from torch_geometric.data.collate import collate
from .base import *


class EPGCNDS(BaseLightningModel):
    r"""An implementation of the EPGCN-DS model from [sun2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/22

    .. [sun2020] Sun, M., *et al.* (2020). `Structure-Based Drug-Drug Interaction Detection via Expressive
       Graph Convolutional Networks and Deep Sets <https://doi.org/10.1609/aaai.v34i10.7236>`_.
       *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(10), 13927–13928.
    """

    def __init__(
        self,
        conf,
        hidden_channels: int = 32,
        middle_channels: int = 16,
        out_channels: int = 1,
    ):
        """Instantiate the EPGCN-DS model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The number of graph convolutional filters.
        :param middle_channels: The number of hidden layer neurons in the last layer.
        :param out_channels: The number of output channels.
        """
        super().__init__(conf)
        ############################################################# 
        self.module    = nn.ModuleDict()
        self.feat_type = conf.model_params.compound_features
        self.cyp_dim   = 10 if conf.model_params.use_cyp_label else 0 
        molecule_channels  = FEAT2DIM[self.feat_type] + self.cyp_dim
        self.module['embed'] = torch.nn.ModuleList([])
        embedding_dimensions = [121, 10, 12, 13, 10, 6, 9, 3, 3]
        for i, d in enumerate(embedding_dimensions):
            self.module['embed'].append(nn.Embedding(d, 320))
        #############################################################

        self.module['gcn_in']  = GCNConv(320, hidden_channels)
        self.module['gcn_out'] = GCNConv(hidden_channels, middle_channels)
        self.readout = global_mean_pool
        self.module['final']   = nn.Sequential(nn.Linear(middle_channels+self.cyp_dim*2, out_channels), nn.Sigmoid())

    def _forward_molecules(self, x, edge_index, batch):
    	features = self.module['gcn_in'](x, edge_index)
    	features = self.module['gcn_out'](features, edge_index)

    	return self.readout(features, batch)

    def forward(self, input_batch):
        input_batch    = load_to_device(input_batch, 'cuda')
        """Run a forward pass of the EPGCN-DS model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted synergy scores.
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
        hidden       = dcomp_latent + fcomp_latent

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