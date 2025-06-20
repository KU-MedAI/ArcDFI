import pickle
import pandas as pd
import numpy as np
from rdkit import Chem 
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS, Recap
from torch_geometric.data.collate import collate
from rdkit.Chem.rdmolops import FastFindRings
RDLogger.DisableLog('rdApp.*')
import os
import sys
import torch
from torch.utils.data import Dataset 
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
import time
import itertools
from collections import defaultdict
import math
from rdkit.Chem.Scaffolds.MurckoScaffold import *
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.BRICS import BRICSDecompose
from functools import reduce
import random
import torch.nn.functional as F
from copy import deepcopy, copy
from collections import namedtuple, Counter
from datetime import datetime
import pdb
import gc
from tqdm import tqdm
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.BRICS import BRICSDecompose
from pmapper.pharmacophore import Pharmacophore as pha
from rdkit.Chem.rdMolDescriptors import *
from torch_geometric.utils import *
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from rdkit.Chem import rdReducedGraphs
import multiprocessing as mp

from typing import Callable, Optional, Tuple, List, Union, Any, Literal

CYPI_LABELS  = ['1A2_Sub', '3A4_Sub', '2C9_Sub', '2C19_Sub', '2D6_Sub',
                '1A2_Inh', '3A4_Inh', '2C9_Inh', '2C19_Inh', '2D6_Inh']

class DatasetBase(Dataset):
    def __init__(self, conf, partition):
        self.data_instances = []

        self.path_dataset   = os.path.join(conf.path.dataset, f'dfi_{conf.dataprep.version}.csv')
        self.path_features  = os.path.join(conf.path.dataset, f'dfi_{conf.dataprep.version}.features')
        self.path_processed = os.path.join(conf.path.dataset, f'processed_{conf.dataprep.version}/')
        self.dataframe      = pd.read_csv(self.path_dataset).set_index('drugfoodpair_id')
        self.dataframe.dropna(subset='drugcompound_smiles', inplace=True)
        self.dataframe.dropna(subset='foodcompound_smiles', inplace=True)
        self.dataframe      = self.dataframe.sample(frac=conf.dataprep.subsample, random_state=911012)
        if conf.dataprep.split == 'random':
            self.dataframe      = self.dataframe[self.dataframe.split==partition]
        elif conf.dataprep.split == 'newdrug':
            self.dataframe      = self.dataframe[self.dataframe.split_newdrug==partition]
        elif conf.dataprep.split == 'newfood':
            self.dataframe      = self.dataframe[self.dataframe.split_newfood==partition]
        else:
            raise

        if not os.path.exists(self.path_processed):
            os.mkdir(self.path_processed)

def batch_extract_to_list(list_batch_dicts: List[dict], key: str):

    return [bdict[key] for bdict in list_batch_dicts]

def stack_and_pad_with(arr_list, max_length=None, padding_idx=0):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    # T = np.zeros((len(arr_list), M, N))
    T = np.full((len(arr_list), M, N), padding_idx)
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))

    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def collate_(list_batch_data: list, padding=0):
    result_dict = dict()
    try:
        result_dict['collated'] = torch.Tensor(list_batch_data)
        result_dict['masks']    = torch.ones((result_dict['collated'].shape[0], result_dict['collated'].shape[1]))
    except Exception as e:
        results                 = stack_and_pad_with(list_batch_data, padding_idx=padding)
        result_dict['collated'] = torch.Tensor(results[0])
        result_dict['masks']    = torch.Tensor(results[1])

    return result_dict

class PubChemFingerprint(MolecularFeaturizer):
    def __init__(self):
        try:
            from rdkit import Chem  # noqa
            import pubchempy as pcp  # noqa
        except ModuleNotFoundError:
            raise ImportError("This class requires PubChemPy to be installed.")

        self.get_pubchem_compounds = pcp.get_compounds

    def _featurize(self, smiles) -> np.ndarray:
        # smiles = Chem.MolToSmiles(datapoint)
        pubchem_compound = self.get_pubchem_compounds(smiles, 'smiles')[0]
        
        return np.asarray(pubchem_compound.cactvs_fingerprint)