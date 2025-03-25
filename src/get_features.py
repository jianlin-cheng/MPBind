import pickle
import numpy as np
from tqdm import tqdm
import os, sys, argparse, datetime

sys.path.append('.')
sys.path.append('../')

import torch
import torch.nn.functional as F

from src.feature_extraction.ProtTrans import get_ProtTrans
from src.feature_extraction.process_structure import get_pdb_xyz, process_dssp, match_dssp
import torch.utils.data as data
import torch_geometric
from torch_geometric.nn import radius_graph
from esm.data import read_fasta


import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

############ Set to your own path! ############
# %%%%%%%%%%% Note: both ESMfold and ProtT5 models have almost 3B parameters

ProtTrans_path = str(root) + "/src/ProtT5/prot_t5_xl_uniref50"
script_path = str(root) + "/src/feature_extraction"
feats_out = str(root)+"/datasets"

def extract_feat(ID_list, seq_list, outpath, gpu, script_path = script_path, ProtTrans_path = ProtTrans_path):
    # Creat the folder to save correspondig embeddings
    for name in ['DSSP', 'ProtTrans']:
        if not os.path.exists(f"{outpath}/{name}"):
            os.makedirs(f"{outpath}/{name}", exist_ok = True)

    # Min-Max is used to normalize the sequence embeddding from ProtTrans
    Min_protrans = torch.tensor(np.load(script_path + "/Min_ProtTrans_repr.npy"), dtype = torch.float32)
    Max_protrans = torch.tensor(np.load(script_path + "/Max_ProtTrans_repr.npy"), dtype = torch.float32)
    get_ProtTrans(ID_list, seq_list, Min_protrans, Max_protrans, ProtTrans_path, outpath, gpu)   # feature 1: shape=[L, 1024], get the sequence embedding by ProtTrans and saved with '.tensor' extension.
    

    print("Processing PDB files...")
    for ID in tqdm(ID_list):
        with open(outpath + "/chain/" + ID + ".pdb", "r") as f:
            X = get_pdb_xyz(f.readlines()) # [L, 5, 3]  # L=the length of sequence, 5 is five element [N, CA, C, O, R], 3 is the 3D-position of (x, y, z)
        torch.save(torch.tensor(X, dtype = torch.float32), outpath + "/chain/" + ID + '.tensor')  # extraction residue positions from esmfold predictions and saved with '.tensor' extension.


    print("Extracting DSSP features...")
    for i in tqdm(range(len(ID_list))):
        ID = ID_list[i]
        seq = seq_list[i]
        
        # print(f"the ID: {ID}\n")
        os.system("{}/mkdssp -i {}/chain/{}.pdb -o {}/DSSP/{}.dssp".format(script_path, outpath, ID, outpath, ID))

        dssp_seq, dssp_matrix = process_dssp("{}/DSSP/{}.dssp".format(outpath, ID))
        if dssp_seq != seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)

        torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), "{}/DSSP/{}.tensor".format(outpath, ID))  # feature 2: shape=[L, 9], get the structural features by DSSP accoding to the predicted pdb.
        os.system("rm {}/DSSP/{}.dssp".format(outpath, ID))

from transformers import T5Tokenizer, T5EncoderModel
import gc
import re

def ProsT5_embedding(ID_list, seq_list, outpath = ""):
    
    # create the output directory to store the embedding results
    outpath = f"{outpath}/ProsT52AA"

    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok = True)
 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
    gc.collect()

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model.float() if device.type=='cpu' else model.half()
    model = model.eval()

    # translate from AA to 3Di (AA-->3Di)
    with torch.no_grad():
        print("\nExtracting ProstT52AA feature extraction !!!")
        for pid, ss in zip(tqdm(ID_list), seq_list):
      
            # prepare your protein sequences/structures as a list.
            # Amino acid sequences are expected to be upper-case ("PRTEINO" below)
            # while 3Di-sequences need to be lower-case.
            PDBid, sequence_examples = [], []
            PDBid.append(pid)
            sequence_examples.append(ss)
            tempss = sequence_examples
           
            # replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
            sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

            # add pre-fixes accordingly. For the translation from AAs to 3Di, you need to prepend "<AA2fold>"
            # The direction of the translation is indicated by two special tokens:
            # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
            # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
            sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                                    for s in sequence_examples
                                ]
            # tokenize sequences and pad up to the longest sequence in the batch
            ids = tokenizer.batch_encode_plus(sequence_examples,
                                    add_special_tokens=True,
                                    padding="longest",
                                    return_tensors='pt').to(device)

            embedding_rpr = model(
                ids.input_ids, 
                attention_mask=ids.attention_mask
            )

            # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:(L+1)]) 
            embeddings = embedding_rpr.last_hidden_state.cpu()

            # print(f"The number of emdbeddings for PDB id {pid}: {len(embeddings)} and its AA' length: {len(ss)}")
            for seq_num in range(len(embeddings)):
                seq_len = len(tempss[seq_num])
                seq_emd = embeddings[seq_num][1:seq_len+1]

                # print(f"The pid: {PDBid[seq_num]} and its embedding shape: {seq_emd.shape} and corresponding sequence length: {seq_len}")
                # print(tempss[seq_num])
                torch.save(seq_emd, f"{outpath}/{PDBid[seq_num]}.tensor")
       
def extract_Alphafold(ID_list, seq_list, outpath, gpu, filename = None, start = 83153, end = 120000, resume = True):
    # max_len = max([len(seq) for seq in seq_list])

    
    max_len = 1800
    chunk_size = 32 if max_len > 1000 else 64

    # write the current sequence info to train_seq.fa for esmfold to predict the 3D structure
    curr_seq = outpath + "/demo_seq.fa"
    if filename:
        curr_seq = outpath + "/" + filename
    
    if not os.path.exists(curr_seq):
    
        cseq = ""
        for pid, idseq in zip(ID_list, seq_list):
            cseq += (">" + pid + "\n" + idseq + "\n")
    
        with open(curr_seq, "w") as f:
            f.write(cseq)
    
    if not ID_list:
        all_sequence = read_fasta(curr_seq)

        ID_list = []
        seq_list = []
        for ff, ss in tqdm(all_sequence):
            ID_list.append(ff)
            seq_list.append(ss)
    
    print(f"the length Of id: {len(ID_list)} and seqence: {len(seq_list)}\n")

    # Min-Max is used to normalize the sequence embeddding from ProtTrans
    ptrans_path = outpath
    Min_protrans = torch.tensor(np.load(script_path + "feature_extraction/Min_ProtTrans_repr.npy"), dtype = torch.float32)
    Max_protrans = torch.tensor(np.load(script_path + "feature_extraction/Max_ProtTrans_repr.npy"), dtype = torch.float32)
    get_ProtTrans(ID_list, seq_list, Min_protrans, Max_protrans, ProtTrans_path, ptrans_path, gpu)   # feature 1: shape=[L, 1024], get the sequence embedding by ProtTrans and saved with '.tensor' extension.
    

    print("Processing PDB files...")
    # ID_list = ['3NSO_A']
    for ID in tqdm(ID_list):
        # print(f"The current pdb: {ID}\n")
        with open(outpath + "/pdb/" + ID + ".pdb", "r") as f:
            X = get_pdb_xyz(f.readlines()) # [L, 5, 3]  # L=the length of sequence, 5 is five element [N, CA, C, O, R], 3 is the 3D-position of (x, y, z)
        torch.save(torch.tensor(X, dtype = torch.float32), outpath + "/pdb/" + ID + '.tensor')  # extraction residue positions from esmfold predictions and saved with '.tensor' extension.


    print("Extracting DSSP features...")
   
    for i in tqdm(range(len(ID_list))):
        ID = ID_list[i]
        seq = seq_list[i]
        
        # print(f"the ID: {ID}\n")
        os.system("{}/feature_extraction/mkdssp -i {}/pdb/{}.pdb -o {}/DSSP/{}.dssp".format(script_path, outpath, ID, outpath, ID))

        
        dssp_seq, dssp_matrix = process_dssp("{}/DSSP/{}.dssp".format(outpath, ID))
        if dssp_seq != seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)

        torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), "{}/DSSP/{}.tensor".format(outpath, ID))  # feature 2: shape=[L, 9], get the structural features by DSSP accoding to the predicted pdb.
        os.system("rm {}/DSSP/{}.dssp".format(outpath, ID))

##############  DataLoader  ##############
class ProteinGraphDataset(data.Dataset):
    def __init__(self, ID_list, outpath, radius=15):
        super(ProteinGraphDataset, self).__init__()
        self.IDs = ID_list
        self.path = outpath
        self.radius = radius
        self.IDs_n = None
        self.label_pth = f'{outpath}/label/'
        self.prot5_pth = f'{outpath}/ProtTrans/'
        self.prostT5_pth = f'{outpath}/ProsT52AA/'
        # print(f"@@@@@ before The length of IDs: {len(self.IDs)}")
        self.__update_selection()
    
    def __update_selection(self):
        # ckeys mapping with keys
        mask = []
        for pp in self.IDs:
            X = torch.load(self.path + "/chain/" + pp + ".tensor")
            dssp_feat = torch.load(self.path + '/DSSP/' + pp + ".tensor")   # [L, 9]
            
            if X.shape[0] == dssp_feat.shape[0]:
                mask.append(True)
            else:
                mask.append(False)
    
        IDs_n = []
        for pp, mm in zip(self.IDs, mask):
            if mm:
                IDs_n.append(pp)
               
        self.IDs_n = IDs_n
            
    def __len__(self): return len(self.IDs_n)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs_n[idx]
        with torch.no_grad():
            X = torch.load(self.path + "/chain/" + name + ".tensor")  # [L, 5, 3]   5: [N, CA, C, O, R]
            y = torch.load(self.label_pth + name + ".tensor")  # [L, 5, 3]   5: [N, CA, C, O, R]
        
            Res_feat = None
            with open(self.path + "/Atomic/" + name +'.resfea', 'rb') as f:
                Res_feat = pickle.load(f)  # this is in the original numpy array format, the should convert to tensor.float() its shape:[L, 3]  
                Res_feat = torch.from_numpy(Res_feat).float()  # [L, 6]

            prottrans_feat = torch.load(self.prot5_pth + name + ".tensor")  # [L, 1024] # here also need update
            prostT5_feat = torch.load(self.prostT5_pth + name + ".tensor").float()  # [L, 1024] # here also need update

            dssp_feat = torch.load(self.path + '/DSSP/' + name + ".tensor")   # [L, 9]
            pre_computed_node_feat = torch.cat([prostT5_feat, prottrans_feat, Res_feat, dssp_feat], dim=-1)  # [L, 2063]

            X_ca = X[:, 1]  # the shape for X_ca is [L, 3]  # L is the length of sequence
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 100, num_workers = 8)
    
        # graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=node_allft, edge_index=edge_index, edge_attr=edge_attr, y=y, num=num_nodes, vel=vel)
        graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index, y=y)
        
        return graph_data


class ProteinGraphData3Di(data.Dataset):
    def __init__(self, ID_list, outpath, radius=15):
        super(ProteinGraphData3Di, self).__init__()
        self.IDs = ID_list
        self.path = outpath
        self.radius = radius
        self.IDs_n = None
        self.prot5_pth = f'{outpath}/ProtTrans/'
        self.prostT5_pth = f'{outpath}/ProsT52AA/'
        # print(f"@@@@@ before The length of IDs: {len(self.IDs)}")
        self.__update_selection()
    
    def __update_selection(self):
        # ckeys mapping with keys
        mask = []
        for pp in self.IDs:
            X = torch.load(self.path + "/chain/" + pp + ".tensor")
            dssp_feat = torch.load(self.path + '/DSSP/' + pp + ".tensor")   # [L, 9]
            
            if X.shape[0] == dssp_feat.shape[0]:
                mask.append(True)
            else:
                mask.append(False)
    
        IDs_n = []
        for pp, mm in zip(self.IDs, mask):
            if mm:
                IDs_n.append(pp)
               
        self.IDs_n = IDs_n
            
    def __len__(self): return len(self.IDs_n)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs_n[idx]
        with torch.no_grad():
            X = torch.load(self.path + "/chain/" + name + ".tensor")  # [L, 5, 3]   5: [N, CA, C, O, R]
        
            Res_feat = None
            with open(self.path + "/Atomic/" + name +'.resfea', 'rb') as f:
                Res_feat = pickle.load(f)  # this is in the original numpy array format, the should convert to tensor.float() its shape:[L, 3]  
                Res_feat = torch.from_numpy(Res_feat).float()  # [L, 6]

            prottrans_feat = torch.load(self.prot5_pth + name + ".tensor")  # [L, 1024] # here also need update
            prostT5_feat = torch.load(self.prostT5_pth + name + ".tensor").float()  # [L, 1024] # here also need update

            dssp_feat = torch.load(self.path + '/DSSP/' + name + ".tensor")   # [L, 9]
            pre_computed_node_feat = torch.cat([prostT5_feat, prottrans_feat, Res_feat, dssp_feat], dim=-1)  # [L, 2063]

            X_ca = X[:, 1]  # the shape for X_ca is [L, 3]  # L is the length of sequence
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 100, num_workers = 8)
    
        # graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=node_allft, edge_index=edge_index, edge_attr=edge_attr, y=y, num=num_nodes, vel=vel)
        graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index)
        
        return graph_data


class ProteinGraphData3DiN(data.Dataset):
    def __init__(self, ID_list, outpath, radius=15):
        super(ProteinGraphData3DiN, self).__init__()
        self.IDs = ID_list
        self.path = outpath
        self.radius = radius
        self.IDs_n = None
        self.prot5_pth = f'{outpath}/ProtTrans/'
        self.prostT5_pth = f'{outpath}/ProsT52AA/'
        # print(f"@@@@@ before The length of IDs: {len(self.IDs)}")
        self.__update_selection()
    
    def __update_selection(self):
        # ckeys mapping with keys
        mask = []
        for pp in self.IDs:
            X = torch.load(self.path + "/chain/" + pp + ".tensor")
            dssp_feat = torch.load(self.path + '/DSSP/' + pp + ".tensor")   # [L, 9]
            
            if X.shape[0] == dssp_feat.shape[0]:
                mask.append(True)
            else:
                mask.append(False)
        IDs_n = []
        for pp, mm in zip(self.IDs, mask):
            if mm:
                IDs_n.append(pp)
               
        self.IDs_n = IDs_n
            
    def __len__(self): return len(self.IDs_n)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs_n[idx]
        with torch.no_grad():
            X = torch.load(self.path + "/chain/" + name + ".tensor")  # [L, 5, 3]   5: [N, CA, C, O, R]

            Res_feat = None  # this version removed the residue level--atomic features (L, 6)
       
            prottrans_feat = torch.load(self.prot5_pth + name + ".tensor")  # [L, 1024] # here also need update
            prostT5_feat = torch.load(self.prostT5_pth + name + ".tensor").float()  # [L, 1024] # here also need update

            dssp_feat = torch.load(self.path + '/DSSP/' + name + ".tensor")   # [L, 9]
            pre_computed_node_feat = torch.cat([prostT5_feat, prottrans_feat, dssp_feat], dim=-1)  # [L, 2057]

            X_ca = X[:, 1]  # the shape for X_ca is [L, 3]  # L is the length of sequence
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 100, num_workers = 8)
    
        # graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=node_allft, edge_index=edge_index, edge_attr=edge_attr, y=y, num=num_nodes, vel=vel)
        graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index)
        
        return graph_data


class ProteinGraphDataAlphafold(data.Dataset):
    def __init__(self, ID_list, seq_list, ypath, outpath, radius=15,prost3di=True):
        super(ProteinGraphDataAlphafold, self).__init__()
        self.IDs = ID_list
        self.Seqs = seq_list
        self.ypath = ypath
        self.path = outpath
        self.radius = radius
        self.IDs_n = None
        self.prot5_pth = "/cluster/pixstor/chengji-lab/yw7bh/alphafold_structures/ProtTrans/"

        # Below is used to add the embedding features from ProstT5 with foldseek
        self.prost3di = prost3di
        
        if self.prost3di:
            self.prostT5_pth = "/cluster/pixstor/chengji-lab/yw7bh/alphafold_structures/ProstT53Di/"
        else:
            self.prostT5_pth =  "/cluster/pixstor/chengji-lab/yw7bh/alphafold_structures/ProstT5AA/"

        # print(f"@@@@@ before The length of IDs: {len(self.IDs)}")
        self.__update_selection()
    
    def __update_selection(self):
        # ckeys mapping with keys
        mask = []
        for pp in self.IDs:
            X = torch.load(self.path + "/pdb/" + pp + ".tensor")
            dssp_feat = torch.load(self.path + '/DSSP/' + pp + ".tensor")   # [L, 9]
            
            if X.shape[0] == dssp_feat.shape[0]:
                mask.append(True)
            else:
                mask.append(False)
        IDs_n = []
        for pp, mm in zip(self.IDs, mask):
            if mm:
                IDs_n.append(pp)
               
        self.IDs_n = IDs_n
            
    def __len__(self): return len(self.IDs_n)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs_n[idx]
        with torch.no_grad():
            X = torch.load(self.path + "/pdb/" + name + ".tensor")  # [L, 5, 3]   5: [N, CA, C, O, R]
           
            Res_feat = None
            with open(self.path + "/psepos_SC/" + name +'.resfea', 'rb') as f:
                Res_feat = pickle.load(f)  # this is in the original numpy array format, the should convert to tensor.float() its shape:[L, 3]  
                Res_feat = torch.from_numpy(Res_feat).float()  # [L, 6]

            prottrans_feat = torch.load(self.prot5_pth + name + ".tensor")  # [L, 1024] # here also need update
            prostT5_feat = torch.load(self.prostT5_pth + name + ".tensor").float()  # [L, 1024] # here also need update

            dssp_feat = torch.load(self.path + '/DSSP/' + name + ".tensor")   # [L, 9]
            pre_computed_node_feat = torch.cat([prostT5_feat, prottrans_feat, Res_feat, dssp_feat], dim=-1)  # [L, 2063]

            X_ca = X[:, 1]  # the shape for X_ca is [L, 3]  # L is the length of sequence
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 100, num_workers = 8)

        # graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=node_allft, edge_index=edge_index, edge_attr=edge_attr, y=y, num=num_nodes, vel=vel)
        graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index)
        
        return graph_data

##############  Geometric Featurizer  ##############
def get_geo_feat(X, edge_index):
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)  # dim = [N, 12]
    node_dist, edge_dist = _get_distance(X, edge_index) # Node=[N, 160], Ed = [E, 400]
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)   # [N, 184]
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1) # [E, 450]

    return geo_node_feat, geo_edge_feat


def _positional_embeddings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]

    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16] == [N, 160]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16] == [E, 400]

    return node_dist, edge_dist

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]=[L, 12]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3] # slightly improve performance
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]=[E, 30]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [E,3,3]
        Q [E,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q

if __name__ == "__main__":
    '''
    x = torch.randn(8, 5, 3)
    x_ca = x[:, 1]
    edge_index = radius_graph(x_ca, r=1.1, loop=True, max_num_neighbors = 1000, num_workers = 8)
    ndf, egf = get_geo_feat(x, edge_index)
    print(ndf.shape, egf.shape)
    '''
    for dir_name in ["pdb", "ProtTrans", "DSSP", "pred"]:
        os.makedirs(feats_out + "/" + dir_name, exist_ok = True)
