import re
# import sys
import h5py
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm
import pandas as pd
import os
import sys
sys.path.append('.')
sys.path.append('../')

from src.structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits
from src.data_encoding import config_encoding, encode_structure, encode_features, extract_topology, extract_all_contacts
from src.dataset import StructuresDataset, save_data

pt.multiprocessing.set_sharing_strategy('file_system')


config_dataset = {
    # parameters
    "r_thr": 5.0,  # Angstroms
    "max_num_atoms": 1024*8,
    "max_num_nn": 64,
    "molecule_ids": np.array([
        'GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG', 'PHE', 'TYR', 'ILE',
        'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP', 'MET', 'CYS', 'A', 'U', 'G', 'C', 'DA',
        'DT', 'DG', 'DC', 'MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE',
        'NI', 'SR', 'BR', 'CO', 'HG', 'SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT',
        'BMA', 'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP', 'FUC', 'FES',
        'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B', 'AMP', 'NDP', 'SAH', 'OXY', 'PLM',
        'CLR', 'CDL', 'RET'
    ]),  # total 79 types according to the paper

    # input PDB filepaths
    "pdb_filepaths": glob(f"{$PWD}/input_pdb_file_path/*.pdb[0-9]*.gz"),  
    
    # output filepath
    "dataset_filepath": f"{$PWD}/output_h5_file_patH/contacts_rr5A_64nn_8192_wat.h5",

    #PDB relase date filepath
    "release_date_path":f"{$PWD}/pdb_release_info_path/release_date.csv"
}


def contacts_types(s0, M0, s1, M1, ids, molecule_ids, device=pt.device("cpu")):
    """
    Compute contact types between two subunits
    
    Parameters:
    ----------
    s0: subunit 0
    M0: subunit 0's atom-residue map, [num_atoms, num_residues]
    s1: subunit 1
    M1: subunit 1's atom-residue map
    ids: contact indices, [num_contacts, 2]
    molecule_ids: molecule types, [79,]
    device: device
    
    Returns:
    -------
    Y: contact map, [num_residues0, num_residues1, num_molecule_types, num_molecule_types]
    T: assembly type fingerprint matrix, []
    """
    # molecule types for s0 and s1, [num_atoms, num_molecule_types] for one-hot encoding
    c0 = pt.from_numpy(s0['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    c1 = pt.from_numpy(s1['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)

    # categorize contacts, [Num_contacts, num_molecule_types, num_molecule_types]
    H = (c1[ids[:,1]].unsqueeze(1) & c0[ids[:,0]].unsqueeze(2))

    # residue indices of contacts, [Num_contacts]
    rids0 = pt.where(M0[ids[:,0]])[1]
    rids1 = pt.where(M1[ids[:,1]])[1]

    # create detailed contact map: automatically remove duplicated atom-atom to residue-residue contacts
    # [num_residues0, num_residues1, num_type_atoms0, num_type_atoms1]
    Y = pt.zeros((M0.shape[1], M1.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool)
    Y[rids0, rids1] = H

    # define assembly type fingerprint matrix
    T = pt.any(pt.any(Y, dim=1), dim=0)
    # print(f" *********** the contact information for two subunits: ---- Y shape: {Y.shape} and y with true: {pt.any(Y)} M0 shape: {M0.shape} and t with true: {pt.any(T)}  H shape: {H.shape} T shape: {T.shape} ids shape: {ids.shape} and r0: {rids0} r1: {rids1.shape}  and c0 shape: {c0.shape} c1 shape: {c1.shape} \n")
    # exit()
    return Y, T


def pack_structure_data(X, qe, qr, qn, M, ids_topk):
    return {
        'X': X.cpu().numpy().astype(np.float32),
        'ids_topk': ids_topk.cpu().numpy().astype(np.uint16),
        'qe':pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),  # find the indices of truth  ones
        'qr':pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),  # find the indices of truth  ones
        'qn':pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),  # find the indices of truth  ones
        'M':pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),   # find the indices of truth  ones
    }, {
        'qe_shape': qe.shape, 'qr_shape': qr.shape, 'qn_shape': qn.shape,
        'M_shape': M.shape,
    }


def pack_contacts_data(Y, T):
    return {
        'Y':pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16),  # find the indices of truth  ones
    }, {
        'Y_shape': Y.shape, 'ctype': T.cpu().numpy(),
    }

# contacts: stores each pair of atoms with corresponding distance
def pack_dataset_items(subunits, contacts, molecule_ids, max_num_nn, device=pt.device("cpu")):
    # prepare storage
    structures_data = {}
    contacts_data = {}

    # extract features and contacts for all subunits with contacts
    for cid0 in contacts:
        # get subunit
        s0 = subunits[cid0]

        # extract features, encode structure and compute topology
        qe0, qr0, qn0 = encode_features(s0)  # one-hot encoding for element, residues and atoms
        X0, M0 = encode_structure(s0, device=device)   # M0 = [num_atoms, num_residues]
        ids0_topk = extract_topology(X0, max_num_nn)[0]
        # print(f"%%%%%%%%%%%%%%%%% the encode structure info X0 shape {X0.shape} qr0 shape {qr0.shape} and M0.shape {M0.shape}")
        # print(M0)
        # exit()

        # store structure data
        structures_data[cid0] = pack_structure_data(X0, qe0, qr0, qn0, M0, ids0_topk)  # atom_position, atom_ele, atom_res, atom_resid_maping, topk_indices

        # prepare storage
        if cid0 not in contacts_data:
            contacts_data[cid0] = {}

        # for all contacting subunits
        for cid1 in contacts[cid0]:
            # prepare storage for swapped interface
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}

            # if contacts not already computed
            if cid1 not in contacts_data[cid0]:
                # get contacting subunit
                s1 = subunits[cid1]

                # encode structure
                X1, M1 = encode_structure(s1, device=device)  # M1 = [num_atoms, num_residues]

                # nonzero not supported for array with more than I_MAX elements
                # in fact below two situations are the same
                if (M0.shape[1] * M1.shape[1] * (molecule_ids.shape[0]**2)) > 2e9:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]['ids'].cpu()
                    Y, T = contacts_types(s0, M0.cpu(), s1, M1.cpu(), ctc_ids, molecule_ids, device=pt.device("cpu"))
                else:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]['ids'].to(device)
                    Y, T = contacts_types(s0, M0.to(device), s1, M1.to(device), ctc_ids, molecule_ids, device=device)

                # if has contacts of compatible type
                if pt.any(Y):
                    # store contacts data
                    contacts_data[cid0][cid1] = pack_contacts_data(Y, T)
                    contacts_data[cid1][cid0] = pack_contacts_data(Y.permute(1,0,3,2), T.transpose(0,1))

                # clear cuda cache
                pt.cuda.empty_cache()

    return structures_data, contacts_data


def store_dataset_items(hf, pdbid, bid, structures_data, contacts_data):
    # metadata storage
    metadata_l = []

    # for all subunits with contacts
    for cid0 in contacts_data:
        # define store key
        key = f"{pdbid.upper()[1:3]}/{pdbid.upper()}/{bid}/{cid0}" # F8/2F8J/1/A:0/A:0:1

        # save structure data
        hgrp = hf.create_group(f"data/structures/{key}")
        save_data(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])

        # for all contacting subunits
        for cid1 in contacts_data[cid0]:
            # define contacts store key
            ckey = f"{key}/{cid1}"

            # save contacts data
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save_data(hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0])

            # store metadata
            metadata_l.append({
                'key': key,
                'size': (np.max(structures_data[cid0][0]["M"], axis=0)+1).astype(int),
                'ckey': ckey,
                'ctype': contacts_data[cid0][cid1][1]["ctype"],
            })

    return metadata_l

def build_dataset():
    # set up dataset
    dataset = StructuresDataset(config_dataset['pdb_filepaths'], with_preprocessing=False)

    # print(f'-------- The dataset shape is {len(dataset)} -------\n')

    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=16, pin_memory=False, prefetch_factor=4)

    # define device
    device = pt.device("cpu")
    
    # process structure, compute features and write dataset
    pdb_df = pd.DataFrame()
    with h5py.File(config_dataset['dataset_filepath'], 'w', libver='latest') as hf:
        # store dataset encoding
        for key in config_encoding:  # key in  config_encoding = {'std_elements': std_elements, 'std_resnames': std_resnames, 'std_names': std_names}
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)  # change data type from np.array to string_

        # save contact type encoding
        hf["metadata/mids"] = config_dataset['molecule_ids'].astype(np.string_)  # all 79 residues

        # prepare and store all structures
        metadata_l = []
        pbar = tqdm(dataloader)
        for data, pdb_filepath in pbar:
            # check that structure was loaded
            if data is None:
                continue
            (info, structure) = data
            # parse filepath
            m = re.match(r'.*/([a-z0-9]*)\.pdb([0-9]*)\.gz', pdb_filepath)  # () in regex meaning capture and create
            pdbid = m[1] # 
            bid = m[2]
            # print(f'\n ------ {m} the pdb file_path: {pdb_filepath} and  the pdb information as bellow: {m[1]} and {m[2]} ------\n')


            # obtain uniprot labels for each chain. store the pdb_info as an independent
            pdb_df = pd.concat(
                [pdb_df,pd.DataFrame.from_dict({"pdbid":pdbid, **info},orient='index').T],
                ignore_index=True
                )

            
            # check size
            if structure['xyz'].shape[0] >= config_dataset['max_num_atoms']:
                continue

            # process structure
            structure = clean_structure(structure)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # split structure
            subunits = split_by_chain(structure)

            # print(f'---------- after splitting {list(subunits.keys())} ------\n')
            # remove non-atomic structures
            subunits = filter_non_atomic_subunits(subunits)
            # print(list(subunits['A:0']))
            # check not monomer
            if len(subunits) < 2:
                continue

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)

            # extract all contacts from assembly  # radus threshold = 5.0  results all paired atom ids and corresponding distance within each paired subunits
            contacts = extract_all_contacts(subunits, config_dataset['r_thr'], device=device)

            # print(f" *********** the structures_data info: {contacts.keys()} \n")
            # check there are contacts
            if len(contacts) == 0:
                continue

            # pack dataset items
            structures_data, contacts_data = pack_dataset_items(
                subunits, contacts,
                config_dataset['molecule_ids'],
                config_dataset['max_num_nn'], device=device
            )

            # print(f" &&&&&&&&&&& the contact_data info: {contacts_data} \n")

            # store data
            metadata = store_dataset_items(hf, pdbid, bid, structures_data, contacts_data)
            metadata_l.extend(metadata)

            # debug print
            pbar.set_description(f"{metadata_l[-1]['key']}: {metadata_l[-1]['size']}, release date: {info['deposition_date']}")

        # store metadata
        hf['metadata/keys'] = np.array([m['key'] for m in metadata_l]).astype(np.string_)
        hf['metadata/sizes'] = np.array([m['size'] for m in metadata_l])
        hf['metadata/ckeys'] = np.array([m['ckey'] for m in metadata_l]).astype(np.string_)
        hf['metadata/ctypes'] = np.stack(np.where(np.array([m['ctype'] for m in metadata_l])), axis=1).astype(np.uint32)
        pdb_df.to_csv(config_dataset['release_date_path'])
        
if __name__ == "__main__":
    build_dataset()

    