import os
import sys
sys.path.append('.')
sys.path.append('../')

import json
import numpy as np
import pandas as pd
import torch as pt
from tqdm import tqdm
import pickle
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
from src.get_features import ProteinGraphDataset, get_geo_feat, ProteinGraphData3Di, ProteinGraphData3DiN
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

parser = argparse.ArgumentParser(description='Train the ProLEMB with different parameters.')

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

# other parameters need to be defined
parser.add_argument('--folder', type=str, default=f"Train", help='The folder name where the intermidiate file will store')
parser.add_argument('--version', type=str, default="5", help='name the pre-train model version')
parser.add_argument('--nofeature', action='store_true', help="without extract the feature embedding")
parser.add_argument('--track', action='store_true', help="enable the wandb to log the metric online")
args = parser.parse_args()

# where the processed PDB information is stored as an .h5 file
file_path = "/home/yw7bh/data/Projects/FunBench/ProBST/experiments/contacts_rr5A_64nn_8192_wat.h5"

# Common dataloader used to select the pdb files we want to train our model
def setup_dataloader(config_data, sids_selection_filepath, state_test = False):
    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))

    # create dataset
    dataset = Dataset(file_path, features_flags=(True, True, True), seq_info=True)

    # data selection criteria
    m = select_by_sid(dataset, sids_sel) # select by sids
    m &= select_by_max_ba(dataset, config_data['max_ba'])  # select by max assembly count
    m &= (dataset.sizes[:,0] <= config_data['max_size']) # select by max size
    m &= (dataset.sizes[:,1] >= config_data['min_num_res'])  # select by min size
    m &= select_by_interface_types(dataset, config_data['l_types'], np.concatenate(config_data['r_types']))  # select by interface type

    # update dataset selection
    dataset.update_mask(m)

    # set dataset types for labels
    dataset.set_types(config_data['l_types'], config_data['r_types'])

    # define data loader needs to be modified as you want, original batch_size=config_runtime['batch_size']
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=4,  pin_memory=True, prefetch_factor=2)

    return dataloader


# Customized graphdataloader used to load featurized data to train
def set_graphDataloader(pdblist, feats_out = "", batch = 4, shuf = True, drop = True, df_chunk = None):

    pdblist = pdblist

    if df_chunk:
        pdblist =  pdblist[-df_chunk:]
        batch = df_chunk

    dataset = ProteinGraphDataset(pdblist, feats_out)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=shuf, drop_last=drop, num_workers=4, prefetch_factor=2)

    return dataloader

def ff_fusion(device, gdata):
    gdata = gdata.to(device)

    # to get the corresponding features
    name = gdata.name
    # print(f"\nthe size of batch: {name}")

    X = gdata.X
    y = gdata.y
    num_nodes = X.shape[0]
    vel = pt.zeros_like(X[:, 1]).to(device)
    index = gdata.edge_index
    node_feat = gdata.node_feat

    geonf, geoef = get_geo_feat(X, index)  # # geonf =[L, 184], geoef = [E, 450]
    edge_attr = geoef
    node_allft = pt.cat([node_feat, geonf], dim = -1)  # in total, node_features [L, 2247]

    return name, node_allft, X, index, vel, edge_attr, num_nodes, y


def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step):

    _, node_allft, X, edge_index, vel, edge_attr, num_nodes, y = [data for data in batch_data]

    z = model.forward(h=node_allft, x = X, edges=edge_index, vel = vel,  edge_attr=edge_attr,  n_nodes=num_nodes)
    losses = criterion(z, y)  

    return losses, y.detach(), pt.sigmoid(z).detach(), z.detach()

def scoring(eval_results, device=pt.device('cpu')):
    # compute sum losses and scores for each entry
    sum_losses, scores = [], []
    for losses, y, p in eval_results:
        sum_losses.append(pt.sum(losses, dim=0))
        scores.append(bc_scoring(y, p))

    # average scores
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()

    # pack scores
    scores = {'loss': float(np.sum(m_losses))}
    for i in range(m_losses.shape[0]):
        scores[f'{i}/loss'] = m_losses[i]
        for j in range(m_scores.shape[0]):
            scores[f'{i}/{bc_score_names[j]}'] = m_scores[j,i]

    return scores


def logging(logger, writer, scores, global_step, pos_ratios, step_type, track = False):
    # debug print
    pr_str = ', '.join([f"{r:.4f}" for r in pos_ratios])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, pos_ratios=[{pr_str}]")

    # store statistics
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    logger.store(**summary_stats)

    # detailed information
    if track:
        metrics = {k:scores[k] for k in scores if not np.isnan(scores[k])}
        metrics.update({"global_step": global_step})

        # update metrics.keys with prefix name=step_type
        metrics_n = {}
        for k, v in metrics.items():
            metrics_n[f"{step_type}/{k}"] = v

        writer.log(metrics_n)

    # debug print
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))


def log_minloss(log_dir, log_name, **kwargs):
    log_path = os.path.join(log_dir, log_name+'.dat')
    s = pd.Series(kwargs)

    # update log file
    with open(log_path, 'a') as fs:
        fs.write(s.to_json()+'\n')


def train(config_data,  config_runtime, output_path, track = args.track, extract = args.nofeature, version = "5", pdb_path = ""):
    
    # Name model with corresponding version
    weight_path = f"model_{version}"
    # Create logger
    logger = Logger(output_path, f'train_{weight_path}')

    # print configuration
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)

    # define device
    device = pt.device(config_runtime['device'])

    # create model
    model = ClofNet3Di(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    if track:
        pj_name = weight_path
        wandb.login()
        wandb.init(project=f"ProLEMB_{pj_name}", name=f'protein_binding_{pj_name}', resume=True)
        writer = wandb
    else:
        writer = None


    # debug print
    logger.print(">>> Model")
    # logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")

    # define where to store the updated and best models during training
    model_filepath = os.path.join(str(root)+"/weight", f'{weight_path}_ckpt.pt')
    best_filepath = os.path.join(str(root)+"/weight", f'{weight_path}.pt')
    
    min_dir = output_path
    min_name = "min_"+weight_path
  
    # reload model if configured
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath))
        # get last global step
        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
        # dynamic positive weight
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
        
        # To load the minloss from the training history
        min_path = os.path.join(min_dir, min_name+'.dat')
        min_loss =  float(np.array(json.loads([l for l in open(min_path, 'r')][-1])['loss']))
    else:
        # starting global step
        global_step = 0
        # dynamic positive weight
        pos_ratios = 0.5*pt.ones(len(config_data['r_types']), dtype=pt.float).to(device)

        # intialize the min_loss to track
        min_loss = 1e9 


    # setup dataloaders

    dataloader_train = setup_dataloader(config_data, config_data['train_selection_filepath'], state_test=False)
    dataloader_test = setup_dataloader(config_data, config_data['test_selection_filepath'], state_test=False)  # # data_id and data_seq are the list containing str, while y is tensor with shape(batch_size, num_resid, num_type)
 
    # debug print
    logger.print(f"> training data size: {len(dataloader_train)}")
   
    # check the gpu id
    # Define device
    _, cuda_id = (pt.device("cuda"), pt.cuda.current_device()) if pt.cuda.is_available() else (pt.device("cpu"), None)
    
    # bellow is used to extract features 
    # Create the folder for the splited pdb
    splitPDB_path = f'{pdb_path}/chain'
    if not os.path.exists(splitPDB_path):
        os.makedirs(splitPDB_path, exist_ok = True)
    
    label_path = f'{pdb_path}/label'
    if not os.path.exists(label_path):
        os.makedirs(label_path, exist_ok = True)

    if not extract:
        logger.print(">>> start extracting the features")
        all_data = [dataloader_train, dataloader_test]
        ID_list = {0: [], 1: []}  
        for idx, data_loader in enumerate(all_data):
            for batch_data in tqdm(data_loader):  
                try:
                    pdb_id, new_seq, y, cid, filter_struc = batch_data
                    save_pdb({cid: filter_struc}, f'{splitPDB_path}/{pdb_id}.pdb')  # where to save the processed pdb files

                    # Save the label
                    pt.save(y, f'{label_path}/{pdb_id}.tensor')
            
                    # Below is used to generate corresponding features for each chain
                    # First generate the ProtTrans, DSSP, and process the splitted PDB chain
                    extract_feat(ID_list = [pdb_id], seq_list = [new_seq], outpath = pdb_path, gpu = cuda_id, script_path = script_path, ProtTrans_path = ProtTrans_path)

                    # Second generate the Prost5 AA ambeeding
                    ProsT5_embedding(ID_list = [pdb_id], seq_list = [new_seq], outpath = pdb_path)
            
                    # Third get the atomic residue feature from the splitted PDB files
                    PDBFeature(query_id = pdb_id,  PDB_chain_dir = splitPDB_path, results_dir =  f'{pdb_path}/Atomic')
                    atomic_feat = PDBResidueFeature(query_path = f'{pdb_path}/Atomic', query_id = pdb_id)
                    ID_list[idx].append(pdb_id)
                except:
                    pass

        with open(f"{pdb_path}/data_info.pickle", "wb") as f:
            pickle.dump(ID_list, f)
        
        logger.print(">>> finsh feature extraction !!!")
    
    else:
        with open(f"{pdb_path}/data_info.pickle", "rb") as f:
            ID_list = pickle.load(f)
    
    # Create the dataloader with all features information included
    dataloader_train = None
    dataloader_train = set_graphDataloader(ID_list[0], feats_out = pdb_path, batch = 64, shuf = True, drop = False, df_chunk = None)

    dataloader_test = None
    dataloader_test = set_graphDataloader(ID_list[1], feats_out = pdb_path, batch = 64, shuf = True, drop = False, df_chunk = None)

    # debug print
    logger.print(">>> Starting training")

    # send model to device
    model = model.to(device)

    # define losses functions
    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])

    # restart timer
    logger.restart_timer()
   
    # start training
    for epoch in range(1, args.epochs):
        # train mode
        model = model.train()

        # train model
        train_results = []
        out_z = []
        sig_z = []
        lab = []
        all_loss = []

        i = 0
        for batch_train_data in tqdm(dataloader_train):
            # global step

            # to check whether it fits our require
            batch_data = ff_fusion(device, batch_train_data)
            global_step += 1

            i = i + 1
            # set gradient to zero
            optimizer.zero_grad()

            # forward propagation
            losses, y, p, z = eval_step(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

            # backward propagation
            loss = pt.sum(losses)
            loss.backward()

            # optimization step
            optimizer.step()

            # store evaluation results
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])
           
            # log step
            if (global_step+1) % config_runtime["log_step"] == 0:
            # process evaluation results
                with pt.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    train_results = []

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "train_"+weight_path, track = track)

                    # save model checkpoint
                    pt.save(model.state_dict(), model_filepath)


            # evaluation step
            if (global_step+1) % config_runtime["eval_step"] == 0:
                # evaluation mode
                model = model.eval()

                with pt.no_grad():
                    # evaluate model
                    test_results = []
                    for step_te, batch_test_data in enumerate(dataloader_test):

                    # to check whether it fits our require
                        batch_data = ff_fusion(device, batch_test_data)
                
                        # forward propagation
                        losses, y, p, z = eval_step(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

                        # store evaluation results
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # stop evaluating
                        if step_te >= config_runtime['eval_size']:
                            break

                    # scores evaluation results
                    scores = scoring(test_results, device=device)

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "valid_"+weight_path, track = track)

                    # save model and update min loss
                    if min_loss >= scores['loss']:
                        # update min loss
                        min_loss = scores['loss']
                        ss = {"loss": min_loss}
                        
                        #save the corresponding min_loss in the training history, which are set at beginning
                    
                        log_minloss(min_dir, min_name, **ss)

                        # save best model
                        logger.print("> saving model at {}".format(best_filepath))
                        pt.save(model.state_dict(), best_filepath)

                # back in train mode
                model = model.train()


if __name__ == '__main__':
    # Create correspding fold input and output folders if not exist
    pdb_path = str(root) + f"/{args.folder}"
    if not os.path.exists(pdb_path):
        os.makedirs(pdb_path, exist_ok = True)

    train(config_data,  config_runtime, output_path = str(root)+"/weight", pdb_path = pdb_path, version = args.version)
