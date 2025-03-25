import os
import sys
sys.path.append('.')
sys.path.append('../')

import json
import numpy as np
import pandas as pd
import torch as pt
from tqdm import tqdm
import argparse

import wandb
from src.logger import Logger
from src.scoring import bc_scoring, bc_score_names, nanmean
from config import config_data, config_runtime
from data_loader import Dataset, collate_batch_data
from torch_geometric.loader import DataLoader

from src.dataset import select_by_sid, select_by_max_ba, select_by_interface_types

# package import with ProBST need the attention
from models.clof_seqN import ClofNet3Di
from models.clof_seq import ClofNet3Din
from src.get_features import extract_feat, ProteinGraphDataset, get_geo_feat, ProteinGraphData3Di, ProteinGraphData3DiN

from esm.data import read_fasta
import pickle

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

feats_out = str(root)+"/datasets"
parser = argparse.ArgumentParser(description='Inference the ProLEMB with different parameters.')

# below is about the parameters of optimizer
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-16, help='clamp the output of the coords function if get too large')

# initialize model with the following settings.
parser.add_argument('--epochs', type=int, default=401, help='number of epochs to train (default: 10)')  # default = 100
parser.add_argument('--n_layers', type=int, default=4, help='number of layers for the EGLN')
parser.add_argument('--attention', type=int, default=1, help='attention in the EGNN model')
parser.add_argument('--node_dim', type=int, default=2247, help='Number of node features at input')
parser.add_argument('--edge_dim', type=int, default=450, help='Number of edge features at input')  # edge_dim = [Distance + normed R]  # 4
parser.add_argument('--nf', type=int, default=128, help='Number of hidden features')  # default 128
parser.add_argument('--node_attr', type=int, default=1, help='node_attr or not')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',  help='use tanh')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N', help='normalize_diff')
args = parser.parse_args()

# file_path = "contacts_part_wat.h5"  # old 
file_path = "contacts_rr5A_64nn_8192_wat.h5"

def set_graphDataloader(csv_pth, label_pth, feats_out, batch = 4, shuf = True, drop = True, df_chunk = None, version = ""):

    # read seqence info from pandas csv file
    if os.path.exists(csv_pth): 
        df = pd.read_csv(csv_pth, sep="\t")

        pdblist = df["pdb_ID"].to_list()
        seqslist = df["seq"].to_list()
    else:
        path = "/home/yw7bh/data/Projects/FunBench/ProBST/datasets/construct/binding_typeids.pkl"
        with open(path, "rb") as f:
            pdbinfo = pickle.load(f)
            # print([*pdbinfo][0])
            pdblist = pdbinfo[[*pdbinfo][0]]
            seqslist = []

    if df_chunk:
        pdblist =  pdblist[-df_chunk:]
        seqslist = seqslist[-df_chunk:]
        batch = df_chunk
    
    if version == "3":
        dataset = ProteinGraphData3DiN(pdblist, seqslist, label_pth, feats_out, prost3di = False)
    else:
        dataset = ProteinGraphData3Di(pdblist, seqslist, label_pth, feats_out, prost3di = False)

    dataloader = DataLoader(dataset, batch_size=batch, shuffle=shuf, drop_last=drop, num_workers=4, prefetch_factor=2)

    return dataloader

def ff_fusion(device, gdata):
    gdata = gdata.to(device)

    # to get the corresponding features
    name = gdata.name
    
    X = gdata.X
    num_nodes = X.shape[0]
    vel = pt.zeros_like(X[:, 1]).to(device)
    index = gdata.edge_index
    y = gdata.y
    node_feat = gdata.node_feat

    # print(f"\nthe size of batch: {name} x: {X.shape}, y: {y.shape}")

    geonf, geoef = get_geo_feat(X, index)  # # geonf =[L, 184], geoef = [E, 450]
    edge_attr = geoef
    node_allft = pt.cat([node_feat, geonf], dim = -1)  # in total, node_features [L, 2247]

    return name, node_allft, X, index, vel, edge_attr, num_nodes, y


def compute_scores(y, p, device=pt.device('cpu')):
    scores = [bc_scoring(y, p)]
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()
    index = pd.MultiIndex.from_product([
        bc_score_names,
    ], names=['Metric'])
    columns = ['Protein', 'NA', 'ion', 'Ligand', 'Lipid']
    df = pd.DataFrame(m_scores, index=index, columns=columns)
    return df


def inference(config_data, config_runtime, model_path, weight_name, result_csv_path, extract = [False, False], version = "", predict_path = ""):

    # set the weight path for inference    
    weight_path = weight_name
 
    # define device
    device = pt.device(config_runtime['device'])

    # Instantiate the model
    if version == "3":  # this version without the atomic features (N, 6) for each residue 
        node_dim = args.node_dim - 6
        model = ClofNet3Di(in_node_nf=node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    
    elif version == "n":  # this is normalized version of the model
        model = ClofNet3Din(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    else:
        model = ClofNet3Di(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
 

    # reload model if configured
    model_filepath = os.path.join(model_path, f'{weight_path}.pt')
    print(model_filepath)

    if os.path.isfile(model_filepath):
        model.load_state_dict(pt.load(model_filepath, map_location=device))

 
    # setup dataloaders
    
    label_pth = str(root)+"/experiments/"+"valid_labelN/"
    csv_pth = str(root)+"/experiments/"+"valid"+"_dataNval.txt"
    # csv_pth = ""
    # dataloader_test = []
    dataloader_test = set_graphDataloader(csv_pth, label_pth, feats_out, batch = 1, shuf = False, drop = False, df_chunk = None, version = version)

  
    # check the gpu id
    cpu_gpu = config_runtime['device'].split(":")[0]
    if cpu_gpu == "cpu":
        gpu_id = []
    else:
        gpu_id = config_runtime['device'].split(":")[1]
    
    # bellow is used to extract features
    extract_train, extract_val = extract
    if extract_train:

        ID_tall = []
        seq_tall = []

        extract_feat(ID_tall, seq_tall, feats_out, gpu_id, "construct.fa")


    if extract_val:
        
        ID_vall = []
        seq_vall = []

        extract_feat(ID_vall, seq_vall, feats_out, gpu_id, "valid_seqN.fa")

    # Send model to device
    model = model.eval().to(device)
    
    for batch_test_data in tqdm(dataloader_test):
        batch_data = ff_fusion(device, batch_test_data)
        name, node_allft, X, edge_index, vel, edge_attr, num_nodes, y = [data for data in batch_data]    
        name, = name

        z = model.forward(h=node_allft, x = X, edges=edge_index, vel = vel,  edge_attr=edge_attr,  n_nodes=num_nodes)
        y, p = y.detach(), pt.sigmoid(z).detach()
        
        # Save the predicted result       
        pt.save(p.cpu().float(), f"{predict_path}/{name}.tensor")

    print("Inference is done !!!")
       

if __name__ == '__main__':
    # evaluate the model
    result_path = "test_result"
    if not os.path.exists(f"./{result_path}"):
        os.makedirs(f"./{result_path}", exist_ok = True)
     
    predict_path = f"{result_path}/predict"
    if not os.path.exists(f"./{predict_path}"):
        os.makedirs(f"./{predict_path}", exist_ok = True)

    version = "2"  # best model: version is 2


    result_csv_path = f"./{result_path}/test_v{version}.csv"
    weight_name = f"model_v{version}"
    weight_path = str(root)+"/weight"

    inference(config_data,  config_runtime, model_path = weight_path, weight_name = weight_name, result_csv_path = result_csv_path, version = version, predict_path = predict_path)
    
