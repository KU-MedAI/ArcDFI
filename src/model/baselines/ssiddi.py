import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import from_smiles
from torch_geometric.data.collate import collate
from .base import *

class EmbeddingLayer(torch.nn.Module):
    """Attention layer."""

    def __init__(self, feature_number: int):
        """Initialize the relational embedding layer.

        :param feature_number: Number of features.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(feature_number, feature_number))
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(
        self,
        left_representations,
        right_representations,
        alpha_scores,
    ):
        """
        Make a forward pass with the drug representations.

        :param left_representations: Left side drug representations.
        :param right_representations: Right side drug representations.
        :param alpha_scores: Attention scores.
        :returns: Positive label scores vector.
        """
        attention = torch.nn.functional.normalize(self.weights, dim=-1)
        left_representations = torch.nn.functional.normalize(left_representations, dim=-1)
        right_representations = torch.nn.functional.normalize(right_representations, dim=-1)
        attention = attention.view(-1, self.weights.shape[0], self.weights.shape[1])
        scores = alpha_scores * (left_representations @ attention @ right_representations.transpose(-2, -1))
        scores = scores.sum(dim=(-2, -1)).view(-1, 1)
        return scores


class DrugDrugAttentionLayer(torch.nn.Module):
    """Co-attention layer for drug pairs."""

    def __init__(self, feature_number: int):
        """Initialize the co-attention layer.

        :param feature_number: Number of input features.
        """
        super().__init__()
        self.weight_query = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.weight_key = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.bias = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.attention = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.weight_query)
        torch.nn.init.xavier_uniform_(self.weight_key)
        torch.nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        torch.nn.init.xavier_uniform_(self.attention.view(*self.attention.shape, -1))

    def forward(self, left_representations, right_representations):
        """Make a forward pass with the co-attention calculation.

        :param left_representations: Matrix of left hand side representations.
        :param right_representations: Matrix of right hand side representations.
        :returns: Attention scores.
        """
        keys = left_representations @ self.weight_key
        queries = right_representations @ self.weight_query
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        attentions = self.tanh(e_activations) @ self.attention
        return attentions


class SSIDDIBlock(torch.nn.Module):
    """SSIDDI Block with convolution and pooling."""

    def __init__(self, head_number: int, in_channels: int, out_channels: int):
        """Initialize an SSI-DDI Block.

        :param head_number: Number of attention heads.
        :param in_channels: Number of input channels.
        :param out_channels: Number of convolutional filters.
        """
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=head_number, concat=False)
        self.readout = global_mean_pool

    def forward(self, x, edge_index, batch):
        """Make a forward pass.

        :param molecules: A batch of graphs.
        :returns: The molecules with updated atom states and the pooled representations.
        """
        h_nodes  = self.conv(x, edge_index)
        h_graphs = self.readout(h_nodes, batch)
        return h_nodes, h_graphs


class SSIDDI(BaseLightningModel):
    """An implementation of the SSI-DDI model from [nyamabo2021]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/11

    .. [nyamabo2021] Nyamabo, A. K., *et al.* (2021). `SSI–DDI: substructure–substructure interactions
       for drug–drug interaction prediction <https://doi.org/10.1093/bib/bbab133>`_.
       *Briefings in Bioinformatics*, 22(6).
    """

    def __init__(
        self,
        conf,
        hidden_channels=(32, 32),
        head_number=(2, 2),
    ):
        """Instantiate the SSI-DDI model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The list of neurons for each hidden layer block.
        :param head_number: The number of attention heads in each block.
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
        
        molecule_channels         = 320
        self.module['first_norm'] = nn.LayerNorm(molecule_channels)
        self.module['blocks']     = nn.ModuleList([])
        self.module['net_norms']  = nn.ModuleList([])

        channels = molecule_channels
        for hidden_channel, head_number in zip(hidden_channels, head_number):  # noqa: B020
            self.module['blocks'].append(SSIDDIBlock(head_number, channels, hidden_channel))
            self.module['net_norms'].append(nn.LayerNorm(hidden_channel))
            channels = hidden_channel

        self.module['co_attention']  = DrugDrugAttentionLayer(channels)
        self.module['rel_embedding'] = EmbeddingLayer(channels)
        self.module['final']         = nn.Sigmoid()

    def _forward_molecules(self, x, edge_index, batch):
        h = self.module['first_norm'](x)
        representation = []
        for block, net_norm in zip(self.module['blocks'], self.module['net_norms']):
            h, h_pooled = block(h, edge_index, batch)
            representation.append(h_pooled)
            h = torch.nn.functional.elu(net_norm(h))
        return torch.stack(representation, dim=-2)

    def forward(self, input_batch):
        input_batch    = load_to_device(input_batch, 'cuda')
        """Run a forward pass of the SSI-DDI model.

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

        dcomp_features = self._forward_molecules(dcomp_features, dcomp_batch.edge_index, dcomp_batch.batch)
        fcomp_features = self._forward_molecules(fcomp_features, fcomp_batch.edge_index, fcomp_batch.batch)

        attentions     = self.module['co_attention'](dcomp_features, fcomp_features)
        combined       = self.module['rel_embedding'](dcomp_features, fcomp_features, attentions)

        dfi_predicted = self.module['final'](combined)

        # Wrap Up the Results
        output_batch  = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                             fcomp_smiles=input_batch['fcomp_smiles'],
                             y_dfi=input_batch['y_dfi_label'],
                             yhat_dfi=dfi_predicted.reshape(-1))
        return output_batch