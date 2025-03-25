import os
import sys
sys.path.append('.')
sys.path.append('../')

import numpy as np
import pandas as pd
import torch as pt
from tqdm import tqdm
import argparse
from glob import glob

from data_loader import Dataset, collate_batch_data
from torch_geometric.loader import DataLoader

# Package import with ProLEMB needs an eye
from models.clof_seqN import ClofNet3Di
from models.clof_seq import ClofNet3Din
from src.get_features import extract_feat, get_geo_feat, ProteinGraphData3Di, ProteinGraphData3DiN
from src.dataset import select_by_sid, select_by_max_ba, select_by_interface_types, StructuresDataset, collate_batch_features, update_struct, get_seq
from src.structure import encode_bfactor, concatenate_chains
from src.structure_io import save_pdb
from src.PDB_resfeature import PDBFeature, PDBResidueFeature   # get the residue features for each PDB structures
from src.get_features import extract_feat, ProsT5_embedding

from esm.data import read_fasta
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


# Below paths' information should be set correctly
ProtTrans_path = str(root) + "/src/ProtT5/prot_t5_xl_uniref50"  
script_path = str(root) + "/src/feature_extraction"    # The folder where the Dssp tool and  Min-Max feature is in the same folder to normaliza the extract feature from ProtTrans 


parser = argparse.ArgumentParser(description='Inference the ProLEMB with different parameters.')

# initialize model with the following settings.
parser.add_argument('--version', type=str, default="2", help='the trained model version selected to predict, default: 2, the best one')
parser.add_argument('--input', type=str, default=f"example", help='The folder name where the pdb file is')
parser.add_argument('--output', type=str, default=f"Prediction", help='The folder name where the predict output is')
parser.add_argument('--n_layers', type=int, default=4, help='number of layers for the EGLN')
parser.add_argument('--attention', type=int, default=1, help='attention in the EGNN model')
parser.add_argument('--node_dim', type=int, default=2247, help='Number of node features at input')
parser.add_argument('--edge_dim', type=int, default=450, help='Number of edge features at input') 
parser.add_argument('--nf', type=int, default=128, help='Number of hidden features') 
parser.add_argument('--node_attr', type=int, default=1, help='node_attr or not')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',  help='use tanh')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N', help='normalize_diff')
args = parser.parse_args()


def set_graphDataloader(ID_list,  feats_out, batch = 4, shuf = True, drop = True, df_chunk = None, version = ""):

    pdblist = ID_list

    if df_chunk:
        pdblist =  pdblist[-df_chunk:]
        seqslist = seqslist[-df_chunk:]
        batch = df_chunk
    
    if version == "3":
        dataset = ProteinGraphData3DiN(pdblist, feats_out)
    else:
        dataset = ProteinGraphData3Di(pdblist, feats_out)

    dataloader = DataLoader(dataset, batch_size=batch, shuffle=shuf, drop_last=drop, num_workers=4, prefetch_factor=2)

    return dataloader

def ff_fusion(device, gdata):
    gdata = gdata.to(device)

    # Get the corresponding features
    name = gdata.name
    X = gdata.X
    num_nodes = X.shape[0]
    vel = pt.zeros_like(X[:, 1]).to(device)
    index = gdata.edge_index
    node_feat = gdata.node_feat

    geonf, geoef = get_geo_feat(X, index)  # geonf =[L, 184], geoef = [E, 450]
    edge_attr = geoef
    node_allft = pt.cat([node_feat, geonf], dim = -1)  # in total, node_features [L, 2247]

    return name, node_allft, X, index, vel, edge_attr, num_nodes

def inference(pdb_path, model_path, weight_name, version = "", predict_path = ""):

    # Define device
    device, cuda_id = (pt.device("cuda"), pt.cuda.current_device()) if pt.cuda.is_available() else (pt.device("cpu"), None)

    # Instantiate the model
    if version == "3":  # This version without the atomic features (N, 6) for each residue 
        node_dim = args.node_dim - 6
        model = ClofNet3Di(in_node_nf=node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    
    elif version == "n":  # This is normalized version of the model
        model = ClofNet3Din(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    else:
        model = ClofNet3Di(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
 

    # reload model if configured
    weight_path = weight_name
    model_filepath = os.path.join(model_path, f'{weight_path}.pt')
    print(model_filepath)

    if os.path.isfile(model_filepath):
        model.load_state_dict(pt.load(model_filepath, map_location=device))

    else:
        raise ValueError("Can not find the trained model, please check the path for it !!!")
 

    # find pdb files and ignore already predicted oins
    pdb_filepaths = glob(os.path.join(pdb_path, "*.pdb"))
    if len(pdb_filepaths) > 1:
        print(f"The original PDB: {pdb_filepaths[0]}")
    else:
        ValueError("Please check YOUR pdb path\n")

    # create dataset loader with preprocessing
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

    # dataloader_test = set_graphDataloader(csv_pth, feats_out, batch = 1, shuf = False, drop = False, df_chunk = None, version = version)    

    # Send model to device
    model = model.eval().to(device)
    
    # Create the folder for the splited pdb
    splitPDB_path = f'{pdb_path}/chain'
    if not os.path.exists(splitPDB_path):
        os.makedirs(splitPDB_path, exist_ok = True)

    for subunits, filepath in tqdm(dataset):
        # concatenate all chains together
        for cid0 in subunits:
            pdb_id = os.path.basename(filepath).split('.')[0]
            new_seq = None
            structure = subunits[cid0]
            
            resid, mask, filt = get_seq(structure['resname'], structure['resid'])
            new_seq = "".join(resid)

            # Below make sure the chain id exist if the original PDB name without containing the chain id information, 
       
            pdb_id = pdb_id + f"_{cid0.replace(':','_')}"
            
            filter_struc = update_struct(structure, filt)
            filter_struc = encode_bfactor(filter_struc, np.zeros(len(new_seq)))
            save_pdb({cid0: filter_struc}, f'{splitPDB_path}/{pdb_id}.pdb')  # where to save the processed pdb files
            
            # Below is used to generate corresponding features for each chain
            # First generate the ProtTrans, DSSP, and process the splitted PDB chain
            extract_feat(ID_list = [pdb_id], seq_list = [new_seq], outpath = pdb_path, gpu = cuda_id, script_path = script_path, ProtTrans_path = ProtTrans_path)

            # Second generate the Prost5 AA ambeeding
            ProsT5_embedding(ID_list = [pdb_id], seq_list = [new_seq], outpath = pdb_path)
            
            # Third get the atomic residue feature from the splitted PDB files
            PDBFeature(query_id = pdb_id,  PDB_chain_dir = splitPDB_path, results_dir =  f'{pdb_path}/Atomic')
            atomic_feat = PDBResidueFeature(query_path = f'{pdb_path}/Atomic', query_id = pdb_id)
            
            # Set dataset with corresponding model version
            items = set_graphDataloader(ID_list = [pdb_id], feats_out = pdb_path, batch = 1, shuf = False, drop = False, df_chunk = None, version = version)
            items, = items

            # Combine different kinds features to pass the corresponding version
            batch_data = ff_fusion(device, items)
            _, node_allft, X, edge_index, vel, edge_attr, num_nodes = [data for data in batch_data]    

            # Make predictions based on corresponding feature
            z = model.forward(h=node_allft, x = X, edges=edge_index, vel = vel,  edge_attr=edge_attr,  n_nodes=num_nodes)
            p = pt.sigmoid(z).detach()
        
            # Save the predicted result with its corresponding        
            pt.save(p.cpu().float(), f"{predict_path}/{pdb_id}_v{version}.tensor")

    print("Inference is done !!!")
       

if __name__ == '__main__':
    # Inference by ProLEMB
    version = args.version  # best model: version is 2
    weight_name = f"model_v{version}"
    weight_path = str(root)+"/weight"

    # Create correspding fold input and output folders if not exist
    rawpdb_path = str(root) + f"/{args.input}"
    if not os.path.exists(rawpdb_path):
        os.makedirs(rawpdb_path, exist_ok = True)

    result_path = rawpdb_path + f"/{args.output}"
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok = True)

    # Do the inference
    inference(pdb_path = rawpdb_path, model_path = weight_path, weight_name = weight_name, version = version, predict_path = result_path)
    
