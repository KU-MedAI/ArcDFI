from .base import *



class DatasetCore(DatasetBase):
    def __init__(self, conf, partition):
        super().__init__(conf, partition)

        self.conf = conf
        self.pubchemfp = PubChemFingerprint()

    @property 
    def raw_file_names(self):

        return 

    @property
    def processed_file_names(self):

        return 

    def process(self):
        
        return

    def __getitem__(self, idx):
        
        return self.get(idx)

    def get(self):

        return self.data_instances[idx]

    def len(self):

        return len(self.data_instances)

    def __len__(self):

        return self.len()

    @staticmethod
    def check_compound_sanity(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles) 
            return True if mol else False
        except:
            return False

    @staticmethod
    def create_morgan_fingerprint(smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)

        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024)).reshape(1,-1)

    @staticmethod
    def create_pharma_fingerprint(smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)

        return np.array(Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)).reshape(1,-1)

    @staticmethod
    def create_maccs_fingerprint(smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()

        return np.array(GetMACCSKeysFingerprint(mol)).reshape(1,-1)

    @staticmethod
    def create_erg_fingerprint(smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()

        return np.array(rdReducedGraphs.GetErGFingerprint(mol)).reshape(1,-1)

    def create_pubchem_fingerprint(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()

        return self.pubchemfp._featurize(smiles).reshape(1,-1)

    def create_morgan_word_tokens(self, smiles, mol=None):
        fp = self.create_morgan_fingerprint(smiles, mol)

        return np.nonzero(fp)[1].reshape(-1,1)

    def create_pharma_word_tokens(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

        return np.array(fp.GetOnBits()).reshape(-1,1)

    def create_maccs_word_tokens(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fp = np.array(GetMACCSKeysFingerprint(mol)).reshape(1,-1)

        return np.nonzero(fp)[1].reshape(-1,1)

    def create_erg_word_tokens(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fp = np.array(rdReducedGraphs.GetErGFingerprint(mol)).reshape(1,-1)

        return np.nonzero(fp)[1].reshape(-1,1)

    def create_pubchem_word_tokens(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fp = self.pubchemfp._featurize(smiles).reshape(1,-1)

        return np.nonzero(fp)[1].reshape(-1,1)

    def create_brics_fingerprint_set(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fragments = list(BRICSDecompose(mol))

        return np.vstack([self.create_morgan_fingerprint(f) for f in fragments])

    def create_brics_fingerprint_set_pharma(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fragments = list(BRICSDecompose(mol))

        return np.vstack([self.create_pharma_fingerprint(f) for f in fragments])

    def create_brics_fingerprint_set_maccs(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fragments = list(BRICSDecompose(mol))

        return np.vstack([self.create_maccs_fingerprint(f) for f in fragments])

    def create_brics_fingerprint_set_erg(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fragments = list(BRICSDecompose(mol))

        return np.vstack([self.create_erg_fingerprint(f) for f in fragments])

    def create_brics_fingerprint_set_pubchem(self, smiles, mol=None):
        if mol == None:
            mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache()
        FastFindRings(mol)
        fragments = list(BRICSDecompose(mol))

        return np.vstack([self.create_pubchem_fingerprint(f) for f in fragments])

    def create_torch_geometric_graph_data(self, smiles, mol=None):

        return from_smiles(smiles)

    def get_all_compound_features(self, dcomp_smiles, fcomp_smiles, dcomp_mol=None, fcomp_mol=None):
        try:
            return dict(
                dcomp_morgan_fp=self.create_morgan_fingerprint(dcomp_smiles, dcomp_mol),
                # dcomp_pharma_fp=self.create_pharma_fingerprint(dcomp_smiles, dcomp_mol),
                dcomp_maccs_fp=self.create_maccs_fingerprint(dcomp_smiles, dcomp_mol),
                dcomp_erg_fp=self.create_erg_fingerprint(dcomp_smiles, dcomp_mol),
                # dcomp_pubchem_fp=self.create_pubchem_fingerprint(smiles, mol),
                # dcomp_morgan_words=self.create_morgan_word_tokens(smiles, mol),
                # dcomp_pharma_words=self.create_pharma_word_tokens(smiles, mol),
                # dcomp_maccs_words=self.create_maccs_word_tokens(smiles, mol),
                # dcomp_erg_words=self.create_erg_word_tokens(smiles, mol),
                # dcomp_pubchem_words=self.create_pubchem_word_tokens(smiles, mol),
                # dcomp_brics_morgan_set=self.create_brics_fingerprint_set(smiles, mol),
                # dcomp_brics_pharma_set=self.create_brics_fingerprint_set_pharma(smiles, mol),
                # dcomp_brics_maccs_set=self.create_brics_fingerprint_set_maccs(smiles, mol),
                # dcomp_brics_erg_set=self.create_brics_fingerprint_set_erg(smiles, mol),
                # dcomp_brics_pubchem_set=self.create_brics_fingerprint_set_pubchem(smiles, mol),
                # dcomp_mol_graph=self.create_torch_geometric_graph_data(smiles, mol)
                fcomp_morgan_fp=self.create_morgan_fingerprint(fcomp_smiles, fcomp_mol),
                # fcomp_pharma_fp=self.create_pharma_fingerprint(fcomp_smiles, fcomp_mol),
                fcomp_maccs_fp=self.create_maccs_fingerprint(fcomp_smiles, fcomp_mol),
                fcomp_erg_fp=self.create_erg_fingerprint(fcomp_smiles, fcomp_mol),
                # fcomp_pubchem_fp=self.create_pubchem_fingerprint(smiles, mol),
                # fcomp_morgan_words=self.create_morgan_word_tokens(smiles, mol),
                # fcomp_pharma_words=self.create_pharma_word_tokens(smiles, mol),
                # fcomp_maccs_words=self.create_maccs_word_tokens(smiles, mol),
                # fcomp_erg_words=self.create_erg_word_tokens(smiles, mol),
                # fcomp_pubchem_words=self.create_pubchem_word_tokens(smiles, mol),
                # fcomp_brics_morgan_set=self.create_brics_fingerprint_set(smiles, mol),
                # fcomp_brics_pharma_set=self.create_brics_fingerprint_set_pharma(smiles, mol),
                # fcomp_brics_maccs_set=self.create_brics_fingerprint_set_maccs(smiles, mol),
                # fcomp_brics_erg_set=self.create_brics_fingerprint_set_erg(smiles, mol),
                # fcomp_brics_pubchem_set=self.create_brics_fingerprint_set_pubchem(smiles, mol),
                # fcomp_mol_graph=self.create_torch_geometric_graph_data(smiles, mol)
                )
        except Exception as e:
            return None


class DatasetPreload(DatasetCore):
    def __init__(self, conf, partition):
        super().__init__(conf, partition)
        for pair_id, row in tqdm(self.dataframe.iterrows()):
            if os.path.isfile(os.path.join(self.path_processed, f'{pair_id}.pt')):
                self.data_instances.append(pair_id)
            else:
                if not self.check_compound_sanity(row.drugcompound_smiles):
                    continue
                if not self.check_compound_sanity(row.foodcompound_smiles):
                    continue

                data_instance                      = self.get_all_compound_features(row.drugcompound_smiles, row.foodcompound_smiles)
                data_instance['pair_id']           = pair_id
                data_instance['dcomp_id']          = row.drugcompound_id
                data_instance['fcomp_id']          = row.foodcompound_id
                data_instance['dcomp_smiles']      = row.drugcompound_smiles
                data_instance['fcomp_smiles']      = row.foodcompound_smiles

                def convert_boolean(v):
                    if np.isnan(v):
                        return 0.0
                    if v:
                        return 1.0
                    else:
                        return -1.0

                data_instance['y_dfi_label']      = row.dfi_label
                data_instance['dcomp_dci_labels'] = np.array([convert_boolean(x) for x in row[CYPI_LABELS]]).reshape(1,-1)
                data_instance['dcomp_dci_masks']  = np.array([x != 0.0 for x in data_instance['dcomp_dci_labels']]).reshape(1,-1).astype(float)

                torch.save(data_instance, os.path.join(self.path_processed, f'{pair_id}.pt'))
                self.data_instances.append(pair_id)
        
        

    @property
    def processed_file_names(self):

        return [f'{x}.pt' for x in self.data_instances]

    def get(self, idx):
        data = torch.load(os.path.join(self.path_processed, f'{self.data_instances[idx]}.pt'))

        return data

    def __getitem__(self, idx):

        return self.get(idx)

    def len(self):

        return len(self.data_instances)

    def __len__(self):

        return self.len()


def tokenize(matrix, padding_idx=1024):
    tokenized_indices = [torch.nonzero(row).squeeze(1) for row in matrix]
    max_length        = max(len(indices) for indices in tokenized_indices)
    padded_tensor = torch.full((len(tokenized_indices), max_length), fill_value=padding_idx)

    for i, indices in enumerate(tokenized_indices):
        padded_tensor[i, :len(indices)] = indices

    padding_mask = (padded_tensor != padding_idx).float()

    assert padded_tensor.shape[1] == padding_mask.shape[1]

    return padded_tensor, padding_mask


def collate_fn(batch: list[dict]):
    FEAT2DIM   = dict(morgan=1024,pharma=39972,maccs=167,erg=441,pubchem=881)
    batch_dict = dict()

    for key in batch[0].keys():
        batch_dict[key] = batch_extract_to_list(batch, key)

    batch_dict['dcomp_morgan_fp']    = torch.tensor(data=np.vstack(batch_dict['dcomp_morgan_fp']),  dtype=torch.float32)
    # batch_dict['dcomp_pharma_fp']  = torch.tensor(data=np.vstack(batch_dict['dcomp_pharma_fp']),  dtype=torch.float32)
    batch_dict['dcomp_maccs_fp']     = torch.tensor(data=np.vstack(batch_dict['dcomp_maccs_fp']),   dtype=torch.float32)
    batch_dict['dcomp_erg_fp']       = torch.tensor(data=np.vstack(batch_dict['dcomp_erg_fp']),     dtype=torch.float32)

    batch_dict['fcomp_morgan_fp']    = torch.tensor(data=np.vstack(batch_dict['fcomp_morgan_fp']),  dtype=torch.float32)
    # batch_dict['fcomp_pharma_fp']  = torch.tensor(data=np.vstack(batch_dict['fcomp_pharma_fp']),  dtype=torch.float32)
    batch_dict['fcomp_maccs_fp']     = torch.tensor(data=np.vstack(batch_dict['fcomp_maccs_fp']),   dtype=torch.float32)
    batch_dict['fcomp_erg_fp']       = torch.tensor(data=np.vstack(batch_dict['fcomp_erg_fp']),     dtype=torch.float32)

    batch_dict['dcomp_morgan_words'], batch_dict['dcomp_morgan_masks'] = tokenize(batch_dict['dcomp_morgan_fp'], FEAT2DIM['morgan'])
    batch_dict['dcomp_maccs_words'],  batch_dict['dcomp_maccs_masks']  = tokenize(batch_dict['dcomp_maccs_fp'], FEAT2DIM['maccs'])
    batch_dict['dcomp_erg_words'],    batch_dict['dcomp_erg_masks']    = tokenize(batch_dict['dcomp_erg_fp'], FEAT2DIM['erg'])

    batch_dict['fcomp_morgan_words'], batch_dict['fcomp_morgan_masks'] = tokenize(batch_dict['fcomp_morgan_fp'], FEAT2DIM['morgan'])
    batch_dict['fcomp_maccs_words'],  batch_dict['fcomp_maccs_masks']  = tokenize(batch_dict['fcomp_maccs_fp'], FEAT2DIM['maccs'])
    batch_dict['fcomp_erg_words'],    batch_dict['fcomp_erg_masks']    = tokenize(batch_dict['fcomp_erg_fp'], FEAT2DIM['erg'])

    batch_dict['y_dfi_label']      = torch.tensor(data=batch_dict['y_dfi_label'],                 dtype=torch.float32)
    batch_dict['dcomp_dci_labels'] = torch.tensor(data=np.vstack(batch_dict['dcomp_dci_labels']), dtype=torch.float32)
    batch_dict['dcomp_dci_masks']  = torch.tensor(data=np.vstack(batch_dict['dcomp_dci_masks']),  dtype=torch.float32)
            
    return batch_dict

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    conf = OmegaConf.load(f'./settings.yaml')['dev']    

    dataset = DatasetPreload(conf)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch)
