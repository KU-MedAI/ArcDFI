import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import *

import lightning as L
from torchmetrics.classification import *

from transformers import AutoTokenizer, AutoModel
import esm
from torch_geometric.utils.smiles import *
from torch_geometric.nn import GINEConv
from torch_geometric.data.collate import collate
from torch_geometric.nn.pool import global_mean_pool

from .SetTransformer import *

ATOM_FEATURES = ['atomic_num', 'chirality', 'degree', 
                 'formal_charge', 'num_hs', 'num_radical_electrons',
                 'hybridization', 'is_aromatic', 'is_in_ring']
BOND_FEATURES = ['bond_type', 'stereo', 'is_conjugated']

CYPI_LABELS = ['1A2_Sub', '3A4_Sub', '2C9_Sub', '2C19_Sub', '2D6_Sub',
               '1A2_Inh', '3A4_Inh', '2C9_Inh', '2C19_Inh', '2D6_Inh']

CYPI_FASTAS = {
    '1A2': 'MALSQSVPFSATELLLASAIFCLVFWVLKGLRPRVPKGLKSPPEPWGWPLLGHVLTLGKNPHLALSRMSQRYGDVLQIRIGSTPVLVLSRLDTIRQALVRQGDDFKGRPDLYTSTLITDGQSLTFSTDSGPVWAARRRLAQNALNTFSIASDPASSSSCYLEEHVSKEAKALISRLQELMAGPGHFDPYNQVVVSVANVIGAMCFGQHFPESSDEMLSLVKNTHEFVETASSGNPLDFFPILRYLPNPALQRFKAFNQRFLWFLQKTVQEHYQDFDKNSVRDITGALFKHSKKGPRASGNLIPQEKIVNLVNDIFGAGFDTVTTAISWSLMYLVTKPEIQRKIQKELDTVIGRERRPRLSDRPQLPYLEAFILETFRHSSFLPFTIPHSTTRDTTLNGFYIPKKCCVFVNQWQVNHDPELWEDPSEFRPERFLTADGTAINKPLSEKMMLFGMGKRRCIGEVLAKWEIFLFLAILLQQLEFSVPPGVKVDLTPIYGLTMKHARCEHVQARLRFSIN',

    '3A4': 'MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA',

    '2C9': 'MDSLVVLVLCLSCLLLLSLWRQSSGRGKLPPGPTPLPVIGNILQIGIKDISKSLTNLSKVYGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIFHKRFDYKDQQFLNLMEKLNENIKILSSPWIQICNNFSPIIDYFPGTHNKLLKNVAFMKSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYIDLLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFKKSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVPPFYQLCFIPV',

    '2C19': 'MDPFVVLVLCLSCLLLLSIWRQSSGRGKLPPGPTPLPVIGNILQIDIKDVSKSLTNLSKIYGPVFTLYFGLERMVVLHGYEVVKEALIDLGEEFSGRGHFPLAERANRGFGIVFSNGKRWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIFQKRFDYKDQQFLNLMEKLNENIRIVSTPWIQICNNFPTIIDYFPGTHNKLLKNLAFMESDILEKVKEHQESMDINNPRDFIDCFLIKMEKEKQNQQSEFTIENLVITAADLLGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRGHMPYTDAVVHEVQRYIDLIPTSLPHAVTCDVKFRNYLIPKGTTILTSLTSVLHDNKEFPNPEMFDPRHFLDEGGNFKKSNYFMPFSAGKRICVGEGLARMELFLFLTFILQNFNLKSLIDPKDLDTTPVVNGFASVPPFYQLCFIPV',

    '2D6': 'MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSLQLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVFLARYGPAWREQRRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLLHIPALAGKVLRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAEMEKAKGNPESSFNDENLRIVVADLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPYTTAVIHEVQRFGDIVPLGVTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWEKPFRFHPEHFLDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGVFAFLVSPSPYELCAVPR'}



FEAT2DIM   = dict(morgan=1024,pharma=39972,maccs=167,erg=441,pubchem=881)

def load_to_device(batch_dict, device):
	for k, v in batch_dict.items():
		if isinstance(v, torch.Tensor):
			batch_dict[k] = v.to(device)

	return batch_dict


class SetTransformerBase(nn.Module):
    def __init__(self, 
             hidden_dim:    int, 
             dropout_rate:  float, 
             num_heads:     int, 
             attn_option:   str, 
             same_linear:   bool, 
             norm_method:   str, 
             norm_affine:   bool,
             clean_path:    bool):
        super(SetTransformerBase, self).__init__()


class CrossAttentionRegularization(SetTransformerBase):
    def __init__(self, 
                 hidden_dim:    int, 
                 dropout_rate:  float, 
                 num_heads:     int, 
                 attn_option:   str, 
                 same_linear:   bool, 
                 norm_method:   str, 
                 norm_affine:   bool,
                 clean_path:    bool,
                 num_pseudos:   int):
        super(CrossAttentionRegularization, self).__init__(hidden_dim,
                                                           dropout_rate,
                                                           num_heads,
                                                           attn_option,
                                                           same_linear,
                                                           norm_method,
                                                           norm_affine,
                                                           clean_path)
        pmx_args            = (hidden_dim, num_heads, RFF(hidden_dim), 
                               attn_option, same_linear, norm_method, norm_affine, clean_path)
        self.xrosmab        = PoolingMultiheadCrossAttention(*pmx_args)

        self.pseudo         = nn.Parameter(torch.randn(1, num_pseudos, hidden_dim))
        self.fillmasks      = nn.Parameter(torch.ones(1,num_pseudos), requires_grad=False)

    def forward(self, Q, K, Km):
        Q               = Q.unsqueeze(0)
        K_expanded      = torch.cat([K, self.pseudo.repeat(K.size(0),1,1)],1)
        pseudo_masks    = self.fillmasks.repeat(K.size(0),1)
        Km_expanded     = torch.cat([Km, pseudo_masks], 1) 

        O, attn_weights = self.xrosmab(X=Q.repeat(K_expanded.size(0),1,1), Y=K_expanded, Ym=Km_expanded)
        
        return O, attn_weights


class LossModule(nn.Module):
    def __init__(self, conf):
        super(LossModule, self).__init__()

        self.loss_criterion            = nn.ModuleDict()
        self.loss_criterion['dfi']     = nn.BCELoss()

    def forward(self, **kwargs):
        loss_dict = dict()
        loss_dict['dfi']     = self.loss_criterion['dfi'](kwargs['yhat_dfi'], kwargs['y_dfi'])

        return loss_dict


class LossModuleARC(nn.Module):
    def __init__(self, conf):
        super(LossModuleARC, self).__init__()

        self.loss_criterion            = nn.ModuleDict()
        self.loss_criterion['dfi']     = nn.BCELoss()
        self.loss_criterion['dci_sub'] = nn.BCELoss()
        self.loss_criterion['dci_inh'] = nn.BCELoss()

    def forward(self, **kwargs):
        loss_dict = dict()
        loss_dict['dfi']     = self.loss_criterion['dfi'](kwargs['yhat_dfi'], kwargs['y_dfi'])
        loss_dict['dci_sub'] = self.loss_criterion['dci_sub'](kwargs['yhat_dci_sub'], kwargs['y_dci_sub'])
        loss_dict['dci_inh'] = self.loss_criterion['dci_inh'](kwargs['yhat_dci_inh'], kwargs['y_dci_inh'])

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