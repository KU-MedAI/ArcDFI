from .base import *



class CompoundLanguageModel(nn.Module):
    def __init__(self, conf):
        super(CompoundLanguageModel, self).__init__()

        self.tokenizer           = AutoTokenizer.from_pretrained("gayane/BARTSmiles", add_prefix_space=True)
        self.tokenizer.pad_token = '<pad>'
        self.model               = AutoModel.from_pretrained("gayane/BARTSmiles")

        if conf.model_params.cpdlm.freeze_layers:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, batch_smiles: List[str]):
        inputs  = self.tokenizer(batch_smiles, 
            padding=True, 
            truncation=True,
            max_length=128, 
            return_tensors="pt", 
            return_token_type_ids=False, 
            add_special_tokens=True)

        outputs = self.model(input_ids=inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda()) 

        return outputs.last_hidden_state, inputs.attention_mask.cuda()

class CompoundSubstructureEmbeddingLayer(nn.Module):
    def __init__(self, conf):
        super(CompoundSubstructureEmbeddingLayer, self).__init__()

        hdim       = conf.model_params.hidden_dim
        ftype      = conf.model_params.fingerprint_type
        self.layer = nn.Sequential(
                        nn.Embedding(FEAT2DIM[ftype]+1, hdim*2, padding_idx=FEAT2DIM[ftype]),
                        nn.Linear(hdim*2, hdim),
                        nn.LeakyReLU(),
                        nn.Dropout(conf.model_params.dropout_rate))

    def forward(self, tokens, masks):

        return self.layer(tokens), masks

class CompoundGraphEncoderModel(nn.Module):
    def __init__(self, conf):
        super(CompoundGraphEncoderModel, self).__init__()

        hdim        = conf.model_params.hidden_dim
        self.node_embed = nn.ModuleList()
        self.edge_embed = nn.ModuleList()

        for i, k in enumerate(ATOM_FEATURES):
            self.node_embed.append(nn.Embedding(len(x_map[k])+1, hdim*2))
        for i, k in enumerate(BOND_FEATURES):
            self.edge_embed.append(nn.Embedding(len(e_map[k])+1, hdim*2))
        
        self.model = GINEConv(
            nn.Sequential(nn.Linear(hdim*2, hdim),
                          nn.LeakyReLU(),
                          nn.Dropout(conf.model_params.dropout_rate)
                )
            )

        self.readout = global_mean_pool

    def forward(self, batch_data):
        node_embeddings = []
        for i, embed in enumerate(self.node_embed):
            node_embeddings.append(embed(batch_data.x[:,i]).unsqueeze(2))
        batch_data.x = torch.cat(node_embeddings, dim=2).mean(2)

        edge_embeddings = []
        for i, embed in enumerate(self.edge_embed):
            edge_embeddings.append(embed(batch_data.edge_attr[:,i]).unsqueeze(2))
        batch_data.edge_attr = torch.cat(edge_embeddings, dim=2).mean(2)

        node_updated = self.model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)

        return self.readout(node_updated, batch_data.batch)

class CYP450LanguageModel(nn.Module):
    def __init__(self, conf):
        super(CYP450LanguageModel, self).__init__()

        # self.model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        # self.model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()

        if conf.model_params.cyplm.freeze_layers:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self):
        batch_fastas = [('CYP450-1A2',  CYPI_FASTAS['1A2']), 
                        ('CYP450-3A4',  CYPI_FASTAS['3A4']), 
                        ('CYP450-2C9',  CYPI_FASTAS['2C9']), 
                        ('CYP450-2C19', CYPI_FASTAS['2C19']),
                        ('CYP450-2D6',  CYPI_FASTAS['2D6'])]

        _, _, inputs = self.batch_converter(batch_fastas)
        inputs       = inputs.cuda()
        masks        = torch.where(inputs > 1,1,0).float().cuda()
        outputs      = self.model(inputs, repr_layers=[6], return_contacts=True)

        trashed      = outputs['logits'].sum() + outputs['contacts'].sum() + outputs['attentions'].sum()

        return outputs['representations'][6].sum(1) / masks.sum(1).view(-1,1), trashed

class DCICrossAttentionBlock(nn.Module):
    def __init__(self, conf):
        super(DCICrossAttentionBlock, self).__init__()

        self.block = CrossAttentionRegularization(
                                            hidden_dim=conf.model_params.hidden_dim,
                                            dropout_rate=conf.model_params.dropout_rate,
                                            num_heads=conf.model_params.settf.num_heads,
                                            attn_option=conf.model_params.settf.attn_option,
                                            same_linear=conf.model_params.settf.same_linear,
                                            norm_method=conf.model_params.settf.norm_method,
                                            norm_affine=conf.model_params.settf.norm_affine,
                                            clean_path=conf.model_params.settf.clean_path,
                                            num_pseudos=conf.model_params.settf.num_pseudos)

    def forward(self, query_cyp_embeds, keyval_cpd_embeds, keyval_cpd_masks):

        return self.block(query_cyp_embeds, keyval_cpd_embeds, keyval_cpd_masks)

class DFIPredictionLayer(nn.Module):
    def __init__(self, conf):
        super(DFIPredictionLayer, self).__init__()

        hdim = conf.model_params.hidden_dim
        self.layer = nn.Sequential(
                        nn.Linear(hdim*hdim*2, int((hdim*hdim*2)**0.5)),
                        nn.BatchNorm1d(int((hdim*hdim*2)**0.5)),
                        nn.LeakyReLU(),
                        nn.Dropout(conf.model_params.dropout_rate),
                        nn.Linear(int((hdim*hdim*2)**0.5), hdim*2),
                        nn.LeakyReLU(),
                        nn.Dropout(conf.model_params.dropout_rate),
                        nn.Linear(hdim*2, hdim),
                        nn.BatchNorm1d(hdim),
                        nn.LeakyReLU(),
                        nn.Dropout(conf.model_params.dropout_rate),
                        nn.Linear(hdim, 1),
                        nn.Sigmoid())


    def forward(self, dcomp_pooled, cyp450_pooled_fcomp, fcomp_pooled, cyp450_pooled_dcomp):
        cyp450_fcomp_dcomp = dcomp_pooled.unsqueeze(2) * cyp450_pooled_fcomp.unsqueeze(1)
        b, d1, d2       = cyp450_fcomp_dcomp.size()
        cyp450_fcomp_dcomp = cyp450_fcomp_dcomp.reshape(b, d1*d2)

        cyp450_dcomp_fcomp = fcomp_pooled.unsqueeze(2) * cyp450_pooled_dcomp.unsqueeze(1)
        b, d1, d2       = cyp450_dcomp_fcomp.size()
        cyp450_dcomp_fcomp = cyp450_dcomp_fcomp.reshape(b, d1*d2)

        return self.layer(torch.cat([cyp450_fcomp_dcomp,cyp450_dcomp_fcomp],dim=1))


class Model(BaseLightningModel):
    def __init__(self, conf):
        super(Model, self).__init__(conf)

        # self.cpdlm        = CompoundLanguageModel(conf) # 1024
        self.cpdse        = CompoundSubstructureEmbeddingLayer(conf)
        self.cpdge        = CompoundGraphEncoderModel(conf)

        self.cyplm        = CYP450LanguageModel(conf)   # 320

        # self.lin1         = nn.Sequential(
        #                         nn.Linear(1024, conf.model_params.hidden_dim),
        #                         nn.LeakyReLU(),
        #                         nn.Dropout(conf.model_params.dropout_rate))

        self.lin2         = nn.Sequential(
                                nn.Linear(480,  conf.model_params.hidden_dim),
                                nn.LeakyReLU(),
                                nn.Dropout(conf.model_params.dropout_rate))

        self.dciab        = DCICrossAttentionBlock(conf)

        self.dfi          = DFIPredictionLayer(conf)

        self._num_heads   = conf.model_params.settf.num_heads
        self._num_pseudos = conf.model_params.settf.num_pseudos
        self._fp_type     = conf.model_params.fingerprint_type

        self.train_params = conf.train_params
        self.loss_module  = LossModuleARC(conf)

        self.metric_module = nn.ModuleDict()
        for m in ['train', 'valid', 'test']:
            for n in ['dfi', 'dci_sub', 'dci_inh']:
                self.metric_module[f'{m}/{n}/accuracy']  = BinaryAccuracy()
                self.metric_module[f'{m}/{n}/auroc']     = BinaryAUROC()
                self.metric_module[f'{m}/{n}/f1']        = BinaryF1Score()
                self.metric_module[f'{m}/{n}/precision'] = BinaryPrecision()
                self.metric_module[f'{m}/{n}/recall']    = BinaryRecall()
                self.metric_module[f'{m}/{n}/auprc']     = BinaryAveragePrecision()

    def convert_attnweights_to_probscores(self, attnweights, batch_size):
        sub_scores1 = attnweights[0*batch_size:1*batch_size,:,:-self._num_pseudos].sum(2).unsqueeze(2)
        sub_scores0 = attnweights[0*batch_size:1*batch_size,:,-self._num_pseudos:].sum(2).unsqueeze(2)

        inh_scores1 = attnweights[1*batch_size:2*batch_size,:,:-self._num_pseudos].sum(2).unsqueeze(2)
        inh_scores0 = attnweights[1*batch_size:2*batch_size,:,-self._num_pseudos:].sum(2).unsqueeze(2)

        ind_scores1 = attnweights[2*batch_size:3*batch_size,:,:-self._num_pseudos].sum(2).unsqueeze(2)
        ind_scores0 = attnweights[2*batch_size:3*batch_size,:,-self._num_pseudos:].sum(2).unsqueeze(2)

        return torch.cat((sub_scores0, sub_scores1),dim=2), torch.cat((inh_scores0, inh_scores1),dim=2),      


    def training_step(self, batch, batch_idx):
        output  = self.forward(batch)

        losses  = self.loss_module(**output)
        loss    = self.train_params.loss_coef.dfi*losses['dfi'] + self.train_params.loss_coef.dci*losses['dci_sub'] + self.train_params.loss_coef.dci*losses['dci_inh'] + output['unused']
        for k,v in losses.items(): self.log(f'train/loss/{k}', v.item(), sync_dist=True)
        self.log('train/loss/all', loss.item(), sync_dist=True)
        
        self.metric_module['train/dfi/accuracy'](output['yhat_dfi'], output['y_dfi'].long())
        self.log('train/dfi/accuracy', self.metric_module['train/dfi/accuracy'])
        for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
            self.metric_module[f'train/dfi/{m}'].update(output['yhat_dfi'], output['y_dfi'].long())

        if output['y_dci_sub'].numel() != 0:
            self.metric_module['train/dci_sub/accuracy'](output['yhat_dci_sub'], output['y_dci_sub'].long())
            for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.metric_module[f'train/dci_sub/{m}'].update(output['yhat_dci_sub'], output['y_dci_sub'].long())
        if output['y_dci_inh'].numel() != 0:
            self.metric_module['train/dci_inh/accuracy'](output['yhat_dci_inh'], output['y_dci_inh'].long())
            for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.metric_module[f'train/dci_inh/{m}'].update(output['yhat_dci_inh'], output['y_dci_inh'].long())

        return loss

    def on_train_epoch_end(self):
        for m in ['dfi', 'dci_sub', 'dci_inh']:
            self.metric_module[f'train/{m}/accuracy'].reset()
            for n in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.log(f'train/{m}/{n}', self.metric_module[f'train/{m}/{n}'].compute(), sync_dist=True)


    def validation_step(self, batch, batch_idx):
        output  = self.forward(batch)

        losses  = self.loss_module(**output)
        loss    = self.train_params.loss_coef.dfi*losses['dfi'] + self.train_params.loss_coef.dci*losses['dci_sub'] + self.train_params.loss_coef.dci*losses['dci_inh']
        for k,v in losses.items(): self.log(f'valid/loss/{k}', v.item(), sync_dist=True)
        self.log('valid/loss/all', loss.item(), sync_dist=True)
        
        self.metric_module['valid/dfi/accuracy'](output['yhat_dfi'], output['y_dfi'].long())
        self.log('valid/dfi/accuracy', self.metric_module['valid/dfi/accuracy'])
        for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
            self.metric_module[f'valid/dfi/{m}'].update(output['yhat_dfi'], output['y_dfi'].long())

        if output['y_dci_sub'].numel() != 0:
            self.metric_module['valid/dci_sub/accuracy'](output['yhat_dci_sub'], output['y_dci_sub'].long())
            for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.metric_module[f'valid/dci_sub/{m}'].update(output['yhat_dci_sub'], output['y_dci_sub'].long())
        if output['y_dci_inh'].numel() != 0:
            self.metric_module['valid/dci_inh/accuracy'](output['yhat_dci_inh'], output['y_dci_inh'].long())
            for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.metric_module[f'valid/dci_inh/{m}'].update(output['yhat_dci_inh'], output['y_dci_inh'].long())

        return loss

    def on_validation_epoch_end(self):
        for m in ['dfi', 'dci_sub', 'dci_inh']:
            self.metric_module[f'valid/{m}/accuracy'].reset()
            for n in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.log(f'valid/{m}/{n}', self.metric_module[f'valid/{m}/{n}'].compute(), sync_dist=True)


    def test_step(self, batch, batch_idx):
        output  = self.forward(batch)

        losses  = self.loss_module(**output)
        loss    = self.train_params.loss_coef.dfi*losses['dfi'] + self.train_params.loss_coef.dci*losses['dci_sub'] + self.train_params.loss_coef.dci*losses['dci_inh']
        for k,v in losses.items(): self.log(f'test/loss/{k}', v.item(), sync_dist=True)
        self.log('test/loss/all', loss.item(), sync_dist=True)
        
        self.metric_module['test/dfi/accuracy'](output['yhat_dfi'], output['y_dfi'].long())
        self.log('test/dfi/accuracy', self.metric_module['test/dfi/accuracy'])
        for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
            self.metric_module[f'test/dfi/{m}'].update(output['yhat_dfi'], output['y_dfi'].long())

        if output['y_dci_sub'].numel() != 0:
            self.metric_module['test/dci_sub/accuracy'](output['yhat_dci_sub'], output['y_dci_sub'].long())
            for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.metric_module[f'test/dci_sub/{m}'].update(output['yhat_dci_sub'], output['y_dci_sub'].long())
        if output['y_dci_inh'].numel() != 0:
            self.metric_module['test/dci_inh/accuracy'](output['yhat_dci_inh'], output['y_dci_inh'].long())
            for m in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.metric_module[f'test/dci_inh/{m}'].update(output['yhat_dci_inh'], output['y_dci_inh'].long())
        return loss

    def on_test_epoch_end(self):
        for m in ['dfi', 'dci_sub', 'dci_inh']:
            self.metric_module[f'test/{m}/accuracy'].reset()
            for n in ['auroc', 'f1', 'precision', 'recall', 'auprc']:
                self.log(f'test/{m}/{n}', self.metric_module[f'test/{m}/{n}'].compute(), sync_dist=True)


    def forward(self, input_batch):  
        input_batch  = load_to_device(input_batch, 'cuda')

        assert len(input_batch['dcomp_smiles']) == len(input_batch['fcomp_smiles'])
        batch_size = len(input_batch['dcomp_smiles'])

        list_dcomp_data = [from_smiles(x) for x in input_batch['dcomp_smiles']]
        dcomp_batch = collate(list_dcomp_data[0].__class__, list_dcomp_data, False, True)[0].to(self.device)
        list_fcomp_data = [from_smiles(x) for x in input_batch['fcomp_smiles']]
        fcomp_batch = collate(list_dcomp_data[0].__class__, list_fcomp_data, False, True)[0].to(self.device)

        # dcomp_embeddings, dcomp_masks = self.cpdlm(input_batch['dcomp_smiles']) 
        dcomp_embeddings, dcomp_masks = self.cpdse(input_batch[f'dcomp_{self._fp_type}_words'], input_batch[f'dcomp_{self._fp_type}_masks'])  
        # dcomp_embeddings              = self.lin1(dcomp_embeddings)

        # fcomp_embeddings, fcomp_masks = self.cpdlm(input_batch['fcomp_smiles'])
        fcomp_embeddings, fcomp_masks = self.cpdse(input_batch[f'fcomp_{self._fp_type}_words'], input_batch[f'fcomp_{self._fp_type}_masks'])  
        # fcomp_embeddings              = self.lin1(fcomp_embeddings)

        cyp450_embeddings, unused     = self.cyplm()
        cyp450_embeddings             = self.lin2(cyp450_embeddings)
  
        cyp450_embeddings_dcomp, cyp450_attnweights_dcomp = self.dciab(cyp450_embeddings, dcomp_embeddings, dcomp_masks)
        cyp450_embeddings_fcomp, cyp450_attnweights_fcomp = self.dciab(cyp450_embeddings, fcomp_embeddings, fcomp_masks)

        sub_probscores, inh_probscores = self.convert_attnweights_to_probscores(cyp450_attnweights_dcomp, batch_size)

        # dcomp_pooled                   = dcomp_embeddings.sum(1) / dcomp_masks.sum(1).view(-1,1)
        dcomp_pooled                   = self.cpdge(dcomp_batch)
        cyp450_pooled_fcomp            = cyp450_embeddings_fcomp.mean(1)

        # fcomp_pooled                   = fcomp_embeddings.sum(1) / fcomp_masks.sum(1).view(-1,1)
        fcomp_pooled                   = self.cpdge(fcomp_batch)
        cyp450_pooled_dcomp            = cyp450_embeddings_dcomp.mean(1)

        yhat_dfi_score                 = self.dfi(dcomp_pooled, cyp450_pooled_fcomp, fcomp_pooled, cyp450_pooled_dcomp)

        y_dfi                          = input_batch['y_dfi_label']
        yhat_dfi                       = yhat_dfi_score.reshape(-1) 

        y_dci_sub                      = input_batch['dcomp_dci_labels'][:,:5].reshape(-1)
        y_dci_sub                      = (y_dci_sub + 1)/2
        m_dci_sub                      = (y_dci_sub != 0.5)
        y_dci_sub                      = y_dci_sub[m_dci_sub]
        yhat_dci_sub                   = sub_probscores[:,:,-1].reshape(-1)
        yhat_dci_sub                   = yhat_dci_sub[m_dci_sub]

        y_dci_inh                      = input_batch['dcomp_dci_labels'][:,5:].reshape(-1)
        y_dci_inh                      = (y_dci_inh + 1)/2
        m_dci_inh                      = (y_dci_inh != 0.5)
        y_dci_inh                      = y_dci_inh[m_dci_inh]
        yhat_dci_inh                   = inh_probscores[:,:,-1].reshape(-1)
        yhat_dci_inh                   = yhat_dci_inh[m_dci_inh]

        output_batch                   = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                                              fcomp_smiles=input_batch['fcomp_smiles'],
                                              y_dfi=y_dfi,
                                              yhat_dfi=yhat_dfi,
                                              y_dci_sub=y_dci_sub,
                                              y_dci_inh=y_dci_inh,
                                              yhat_dci_sub=yhat_dci_sub,
                                              yhat_dci_inh=yhat_dci_inh,
                                              unused=unused*0.)

        return output_batch

    @torch.no_grad()
    def infer(self, input_batch):
        input_batch  = load_to_device(input_batch, 'cuda')

        list_dcomp_data = [from_smiles(x) for x in input_batch['dcomp_smiles']]
        dcomp_batch = collate(list_dcomp_data[0].__class__, list_dcomp_data, False, True)[0].to(self.device)
        list_fcomp_data = [from_smiles(x) for x in input_batch['fcomp_smiles']]
        fcomp_batch = collate(list_dcomp_data[0].__class__, list_fcomp_data, False, True)[0].to(self.device)

        # dcomp_embeddings, dcomp_masks = self.cpdlm(input_batch['dcomp_smiles']) 
        dcomp_embeddings, dcomp_masks = self.cpdse(input_batch[f'dcomp_{self._fp_type}_words'], input_batch[f'dcomp_{self._fp_type}_masks'])  
        # dcomp_embeddings              = self.lin1(dcomp_embeddings)

        # fcomp_embeddings, fcomp_masks = self.cpdlm(input_batch['fcomp_smiles'])
        fcomp_embeddings, fcomp_masks = self.cpdse(input_batch[f'fcomp_{self._fp_type}_words'], input_batch[f'fcomp_{self._fp_type}_masks'])  
        # fcomp_embeddings              = self.lin1(fcomp_embeddings)

        cyp450_embeddings, unused     = self.cyplm()
        cyp450_embeddings             = self.lin2(cyp450_embeddings)
  
        cyp450_embeddings_dcomp, cyp450_attnweights_dcomp = self.dciab(cyp450_embeddings, dcomp_embeddings, dcomp_masks)
        cyp450_embeddings_fcomp, cyp450_attnweights_fcomp = self.dciab(cyp450_embeddings, fcomp_embeddings, fcomp_masks)

        # dcomp_pooled                   = dcomp_embeddings.sum(1) / dcomp_masks.sum(1).view(-1,1)
        dcomp_pooled                   = self.cpdge(dcomp_batch)
        cyp450_pooled_fcomp            = cyp450_embeddings_fcomp.mean(1)

        # fcomp_pooled                   = fcomp_embeddings.sum(1) / fcomp_masks.sum(1).view(-1,1)
        fcomp_pooled                   = self.cpdge(fcomp_batch)
        cyp450_pooled_dcomp            = cyp450_embeddings_dcomp.mean(1)

        yhat_dfi_score                 = self.dfi(dcomp_pooled, cyp450_pooled_fcomp, fcomp_pooled, cyp450_pooled_dcomp)

        yhat_dfi                       = yhat_dfi_score.reshape(-1) 


        output_batch                   = dict(dcomp_smiles=input_batch['dcomp_smiles'],
                                              fcomp_smiles=input_batch['fcomp_smiles'],
                                              dcomp_attn_weights=cyp450_attnweights_dcomp,
                                              fcomp_attn_weights=cyp450_attnweights_fcomp,
                                              yhat_dfi=yhat_dfi,
                                              unused=unused*0.)

        return output_batch