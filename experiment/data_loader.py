import h5py
import numpy as np
import torch as pt
import os, sys
sys.path.append('.')
sys.path.append('../')

from src.dataset import load_sparse_mask, collate_batch_features
from src.data_encoding import categ_to_resnames

# for save filtered data in sequence/fasta format or pdb format.
from src.structure import data_to_structure, encode_bfactor
from src.structure_io import save_pdb
from src.get_features import extract_feat


import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

feats_out = str(root)+"/datasets"

def load_interface_labels(hgrp, t0, t1_l):
    # load stored data
    shape = tuple(hgrp.attrs['Y_shape'])
    ids = pt.from_numpy(np.array(hgrp['Y']).astype(np.int64))

    # matching contacts type for receptor and ligand
    y_ctc_r = pt.any((ids[:,2].view(-1,1) == t0), dim=1).view(-1,1)
    y_ctc_l = pt.stack([pt.any((ids[:,3].view(-1,1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = (y_ctc_r & y_ctc_l)
    # print(f"/n************* the testing shape: {shape} and y_ctc_r: {y_ctc_r.shape} y_ctc_l: {y_ctc_l.shape} and ids: {ids} and t0: {t0} **************")

    # save contacts of right type
    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    y[ids[:,0], pt.where(y_ctc)[1]] = True

    return y


def collate_batch_data(batch_data):
    # collate features
    X, ids_topk, q, M = collate_batch_features(batch_data)

    # collate targets
    y = pt.cat([data[4] for data in batch_data])

    return X, ids_topk, q, M, y


class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath, features_flags=(True, False, False), seq_info = False):
        super(Dataset, self).__init__()
        # store dataset filepath

        self.seq_info = seq_info
        self.dataset_filepath = dataset_filepath

        # selected features to select which of the features you want to use
        self.ftrs = [fn for fn, ff in zip(['qe','qr','qn'], features_flags) if ff]

        # preload data
        with h5py.File(dataset_filepath, 'r') as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/sizes"])
            self.ckeys = np.array(hf["metadata/ckeys"]).astype(np.dtype('U'))
            self.ctypes = np.array(hf["metadata/ctypes"])

            # load parameters to reconstruct data
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(np.dtype('U'))   # element name: 29 + 1
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(np.dtype('U'))   # residual name: 28 + 1
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype('U'))   # atom name: 63 + 1
            self.mids = np.array(hf["metadata/mids"]).astype(np.dtype('U'))   # molecular_ids: 79

        # set default selection mask tell you how subunits in total for one structure
        self.m = np.ones(len(self.keys), dtype=bool)

        # prepare ckeys mapping
        self.__update_selection()

        # set default runtime selected interface types for initialization
        self.t0 = pt.arange(self.mids.shape[0])
        self.t1_l = [pt.arange(self.mids.shape[0])]

    def __update_selection(self):
        # ckeys mapping with keys

        self.ckeys_map = {}
        for key, ckey in zip(self.keys[self.m], self.ckeys[self.m]):  # map ckeys within the selected keys by self.m
            if key in self.ckeys_map:
                self.ckeys_map[key].append(ckey)
            else:
                self.ckeys_map[key] = [ckey]

        # keep unique keys
        self.ukeys = list(self.ckeys_map)

    def update_mask(self, m):
        # update mask which helps us choose the subunits we want to study
        self.m &= m

        # update ckeys mapping
        self.__update_selection()

    def set_types(self, l_types, r_types_l):
        self.t0 = pt.from_numpy(np.where(np.isin(self.mids, l_types))[0])
        self.t1_l = [pt.from_numpy(np.where(np.isin(self.mids, r_types))[0]) for r_types in r_types_l]

    def get_largest(self):
        i = np.argmax(self.sizes[:,0] * self.m.astype(int))
        k = np.where(np.isin(self.ukeys, self.keys[i]))[0][0]
        return self[k]

    def __len__(self):
        return len(self.ukeys)

    def __getitem__(self, k):
        # get corresponding interface keys
        key = self.ukeys[k]
        ckeys = self.ckeys_map[key]

        # load data
        with h5py.File(self.dataset_filepath, 'r') as hf:
            # hdf5 group
            hgrp = hf['data/structures/'+key]

            # topology
            X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32)) # [num_atoms, 3]
            M = load_sparse_mask(hgrp, 'M') # [num_atoms, num_residues]
            ids_topk = pt.from_numpy(np.array(hgrp['ids_topk']).astype(np.int64)) # [num_atoms, num_closest_atoms]

            # features  # [num_atoms, num_std_element] with one-hot encoding
            q_l = []
            for fn in self.ftrs:
                q_l.append(load_sparse_mask(hgrp, fn))
            q = pt.cat(q_l, dim=1)

            # interface labels
            y = pt.zeros((M.shape[1], len(self.t1_l)), dtype=pt.bool)

            for ckey in ckeys:

                y |= load_interface_labels(hf['data/contacts/'+ckey], self.t0, self.t1_l)

        new_seq = None
        if self.seq_info and len(self.ftrs) == 3:
            filter_struc = data_to_structure(X, q, M, self.std_elements, self.std_resnames, self.std_names)
            resid, mask, filt = get_seq(filter_struc['resname'], filter_struc['resid'])
            new_seq = "".join(resid)
            pdb_id = cast_pdbname(key)
            y = y[mask, :]
            
            cid = key.split('/')[-1][0]
            filter_struc = update_struct(filter_struc, filt)
            filter_struc = encode_bfactor(filter_struc, np.zeros(y[:, 0].cpu().numpy().shape))
            
            return pdb_id, new_seq, y.float(), cid, filter_struc
        else:
            return key, X, ids_topk, q, M, y.float()

# bellow functions are used to get the corresponding sequences' information
res3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

# pdb_pid_chain_cid  # the format of our output, eg. "116d.pdb1.gz", pdb="116d" string (4 characters), pid = 1 int (1-9), chain: string (A-Z), cid: int (0-9) 
def cast_pdbname(input_name):
    # _, pdb,__, chain = input_name.split('/')
    # return f"{pdb}_{chain.split(':')[0]}"
    _, pdb, ind, chain = input_name.split('/')
    return f"{pdb}_{ind}_{chain.replace(':','_')}"


def get_seq(res, resid):

    seq_fasta = list(res)
    # print(seq_fasta)
    seq_sid = list(resid)
    current_pos = -1000
    seq_filter = []

    # mask is used to mask corresponding label in y

    filt = []
    mask = []
    for seqf, seqs in zip(seq_fasta, seq_sid):

        if current_pos != seqs:
            current_pos = seqs
            if seqf in res3to1:
                seq_filter.append(res3to1[seqf])
                mk = 1
                mask.append(mk)
            else:
                mk = 0
                mask.append(mk)
        # to mask the structure information
        if seqf in res3to1:
            mm = 1
            filt.append(mm)
        else:
            mm = 0
            filt.append(mm)


    mask = pt.tensor(mask).bool()
    filt = pt.tensor(filt).bool()

    return seq_filter, mask, filt


def update_struct(struct, mask):
    ''''
    'xyz': X,
    'name': names,
    'element': elements,
    'resname': resnames,
    'resid': resids,
    'het_flag': het_flags,
    '''
    struct['xyz'] = struct['xyz'][mask, :]
    struct['name'] = struct['name'][mask]
    struct['element'] = struct['element'][mask]
    struct['resname'] = struct['resname'][mask]
    struct['resid'] = struct['resid'][mask]
    struct['het_flag'] = struct['het_flag'][mask]

    # To check whetther two differenr residues corresponds one resid:
    current_pos = -1000
    rnamenew = []
    tname = ""
    for rname, rid in zip(struct['resname'], struct['resid']):
        if rid != current_pos:
            current_pos = rid
            tname = rname
        rnamenew.append(tname)
    
    struct['resname'] = np.array(rnamenew)

    return struct


if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    binder_types = ['protein', 'dna','rna', 'ion', 'ligand', 'lipid']
    metadata = pd.DataFrame(columns=['pdb_id']+binder_types)
    config_data = {
    'l_types': categ_to_resnames['protein'],
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna'],categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ],
    }

    # make sure the directories for extracted features (esm_fold, ProT5, and DSSP) exist
    for dir_name in ["pdb", "ProtTrans", "DSSP", "pred"]:
        os.makedirs(feats_out + "/" + dir_name, exist_ok = True)

    dataset = Dataset('contacts_rr5A_64nn_8192_wat.h5', features_flags=(True, True, True), seq_info=True) # construct_testset/
    dataset.set_types(config_data['l_types'], config_data['r_types'])
