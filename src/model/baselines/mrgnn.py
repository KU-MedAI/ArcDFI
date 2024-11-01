import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import from_smiles
from torch_geometric.data.collate import collate
from .base import *

class MRGNN(BaseLightningModel):
    """An implementation of the MR-GNN model from [xu2019]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/12

    .. [xu2019] Xu, N., *et al.* (2019). `MR-GNN: Multi-resolution and dual graph neural network for
       predicting structured entity interactions <https://doi.org/10.24963/ijcai.2019/551>`_.
       *IJCAI International Joint Conference on Artificial Intelligence*, 2019, 3968–3974.
    """

    def __init__(
        self,
        conf,
        hidden_channels: int = 32,
        middle_channels: int = 16,
        layer_count: int = 4,
        out_channels: int = 1,
    ):
        """Instantiate the MRGNN model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The number of graph convolutional filters.
        :param middle_channels: The number of hidden layer neurons in the last layer.
        :param layer_count: The number of graph convolutional and recurrent blocks.
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
            self.module['embed'].append(nn.Embedding(d, 128))
        #############################################################

        self.module['gcn'] = torch.nn.ModuleList()
        self.module['gcn'].append(GCNConv(128, hidden_channels))
        for _ in range(1, layer_count):
            self.module['gcn'].append(GCNConv(hidden_channels, hidden_channels))
        self.module['border_rnn'] = torch.nn.LSTM(hidden_channels, hidden_channels, 1)
        self.module['middle_rnn'] = torch.nn.LSTM(2 * hidden_channels, 2 * hidden_channels, 1)
        self.readout = global_mean_pool
        self.module['final'] = torch.nn.Sequential(
            # First two are the "bottleneck"
            torch.nn.Linear(6 * hidden_channels + self.cyp_dim*2, middle_channels),
            torch.nn.ReLU(),
            # Second to are the "final"
            torch.nn.Linear(middle_channels, out_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_batch):
        input_batch    = load_to_device(input_batch, 'cuda')
        """Run a forward pass of the MR-GNN model.

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

        gcn_hidden_left = dcomp_features
        gcn_hidden_right = fcomp_features
        left_states, right_states, shared_states = None, None, None
        for conv in self.module['gcn']:
            gcn_hidden_left                     = conv(gcn_hidden_left, dcomp_batch.edge_index)
            rnn_out_left, left_states           = self.module['border_rnn'](gcn_hidden_left[None, :, :], left_states)
            graph_level_left                    = self.readout(gcn_hidden_left, dcomp_batch.batch)
            
            gcn_hidden_right                    = conv(gcn_hidden_right, fcomp_batch.edge_index)
            rnn_out_right, right_states         = self.module['border_rnn'](gcn_hidden_right[None, :, :], right_states)
            graph_level_right                   = self.readout(gcn_hidden_right, fcomp_batch.batch)

            shared_graph_level                  = torch.cat([graph_level_left, graph_level_right], dim=1)
            shared_out, shared_states           = self.module['middle_rnn'](shared_graph_level[None, :, :], shared_states)

        rnn_pooled_left = self.readout(rnn_out_left.squeeze(0),  dcomp_batch.batch)
        rnn_pooled_right = self.readout(rnn_out_right.squeeze(0), fcomp_batch.batch)
        shared_out = shared_out.squeeze(0)
        out = torch.cat([shared_graph_level, shared_out, rnn_pooled_left, rnn_pooled_right], dim=1)
        
        if self.cyp_dim > 0:
            out    = torch.cat([out, 
                                input_batch[f'dcomp_dci_labels'], 
                                input_batch[f'dcomp_dci_labels']],dim=1)

        dfi_predicted = self.module['final'](out)

        # Wrap Up the Results
        output_batch  = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                             fcomp_smiles=input_batch['fcomp_smiles'],
                             y_dfi=input_batch['y_dfi_label'],
                             yhat_dfi=dfi_predicted.reshape(-1))

        return output_batch