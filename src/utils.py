import torch as pt
import numpy as np
import string, os
import torch.utils.data as data
import torch_geometric
from torch_geometric.nn import radius_graph

def unpack_edge_features(X, ids_topk):
    # compute displacement vectors
    R_nn = X[ids_topk-1] - X.unsqueeze(1)
    # compute distance matrix
    D_nn = pt.norm(R_nn, dim=2)
    # mask distances
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()
    # normalize displacement vectors
    R_nn = R_nn / D_nn.unsqueeze(2)

    # get the edge features
    All_nn = pt.cat([D_nn.unsqueeze(2), R_nn], dim = 2)
    st = []
    ed = []

    edge_attr = []
    ids_topk = ids_topk-1
    for i in range(ids_topk.shape[0]):
        ids = [i for _ in range(ids_topk.shape[1])]
        ide = [n for n in ids_topk[i]]
        egt = All_nn[i]
        egt2 = [n for n in egt]

        edge_attr.extend(egt2)
        st.extend(ids)
        ed.extend(ide)
    st = pt.tensor(st).long()
    ed = pt.tensor(ed).long()

    edge_ind = [st.to(X.device), ed.to(X.device)]
    edge_attr = pt.stack(edge_attr,dim = 0).float().to(X.device)

    return edge_ind, edge_attr



class ProteinGraphDataset(data.Dataset):
    def __init__(self, ID_list, outpath, radius=15):
        super(ProteinGraphDataset, self).__init__()
        self.IDs = ID_list
        self.path = outpath
        self.radius = radius

    def __len__(self): return len(self.IDs)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs[idx]
        with pt.no_grad():
            X = pt.load(self.path + "pdb/" + name + ".tensor")  # [L, 5, 3]   5: [N, CA, C, O, R]

            prottrans_feat = pt.load(self.path + "ProtTrans/" + name + ".tensor")  # [L, 1024]
            dssp_feat = pt.load(self.path + 'DSSP/' + name + ".tensor")   # [L, 9]
            pre_computed_node_feat = pt.cat([prottrans_feat, dssp_feat], dim=-1)  # [L, 1033]

            X_ca = X[:, 1]  # the shape for X_ca is [L, 3]  # L is the length of sequence
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 1000, num_workers = 8)

        graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index)
        return graph_data

# =======================================================================================================

MAX_INPUT_SEQ = 1000
MAX_SEQ_LEN = 1500

nn_config = {
    'node_input_dim': 1024 + 9 + 184,
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'layer': 4,
    'augment_eps': 0.1,
    'dropout': 0.2
}


# deal with IDs with different formats: e.g. "sp|P05067|A4_HUMAN Amyloid-beta precursor protein" (UniProt), "7PRW_1|Chains A, B|Glucocorticoid receptor|Homo sapiens" (PDB)
def get_ID(name):
    name = name.split("|")
    ID = "_".join(name[0:min(2, len(name))])
    ID = ID.replace(" ", "_")
    return ID


def remove_non_standard_aa(seq):  # the 20 standard amino acids for protein
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    new_seq = ""
    for aa in seq:
        if aa in standard_aa:
            new_seq += aa
    return new_seq


def process_fasta(fasta_file, outpath):
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID_list.append(get_ID(line[1:-1]))
        elif line[0] in string.ascii_letters:
            seq = line.strip().upper()  # remove any leading and trailing spaces and change all letters to upper
            seq = remove_non_standard_aa(seq)
            seq_list.append(seq[0:min(MAX_SEQ_LEN, len(seq))]) # trim long sequence with limits 1500

    if len(ID_list) == len(seq_list):  # The number of ID_list should be always equal the number of sequence list
        if len(ID_list) > MAX_INPUT_SEQ:
            return 1
        else:
            new_fasta = "" # with processed IDs and seqs
            for i in range(len(ID_list)):
                new_fasta += (">" + ID_list[i] + "\n" + seq_list[i] + "\n")
            with open(outpath + "test_seq.fa", "w") as f:
                f.write(new_fasta)

            return [ID_list, seq_list]
    else:
        return -1


def export_predictions(predictions, seq_list, outpath):
    # original order: ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]
    thresholds = [0.35, 0.47, 0.41, 0.46, 0.73, 0.57, 0.44, 0.65, 0.51, 0.61] # select by maximizing MCC on the cross validation
    index = [2, 3, 1, 0, 8, 9, 4, 5, 6, 7] # switch order to ["DNA", "RNA", "PEP", "PRO", "ATP", "HEME", "ZN", "CA", "MG", "MN"]
    GPSite_binding_scores = {}

    for i, ID in enumerate(predictions):
        seq = seq_list[i]
        preds = predictions[ID]
        norm_preds = []
        binding_scores = [] # protein-level binding scores

        for lig_idx, pred in enumerate(preds):
            threshold = thresholds[lig_idx]

            norm_pred = []
            for score in pred:
                if score > threshold:
                    norm_score = (score - threshold) / (1 - threshold) * 0.5 + 0.5
                else:
                    norm_score = (score / threshold) * 0.5
                norm_pred.append(norm_score)
            norm_preds.append(norm_pred)

            if lig_idx in [4, 5, 6, 7]: # metal ions
                k = 5
            else:
                k = 10
            k = min(k, len(seq))

            idx = np.argpartition(norm_pred, -k)[-k:]
            topk_norm_sores = np.array(norm_pred)[idx]  # to find out the top-k large scores to be future processed later.
            binding_scores.append(topk_norm_sores.mean())

        GPSite_binding_scores[ID] = binding_scores

        # bellow is used to save the prediction into corresponding ID.txt file.
        pred_txt = "No.\tAA\tDNA_binding\tRNA_binding\tPeptide_binding\tProtein_binding\tATP_binding\tHEM_binding\tZN_binding\tCA_binding\tMG_binding\tMN_binding\n"
        for j in range(len(seq)):
            pred_txt += "{}\t{}".format(j+1, seq[j]) # 1-based

            for idx in index:
                norm_score = norm_preds[idx][j]
                pred_txt += "\t{:.3f}".format(norm_score)

            pred_txt += "\n"

        with open("{}/pred/{}.txt".format(outpath, ID), "w") as f:
            f.write(pred_txt)


        # export the predictions to a pdb file (for the visualization in the server)
        '''
        score_lines = []
        for j in range(len(seq)):
            score_line = ""
            for idx in index:
                score = norm_preds[idx][j]
                score = "{:.2f}".format(score * 100)
                score = " " * (6 - len(score)) + score
                score_line += score
            score_lines.append(score_line)

        with open("{}/pdb/{}.pdb".format(outpath, ID), "r") as f:
            lines = f.readlines()

        current_pos = -1
        new_pdb = ""
        for line in lines:
            if line[0:4] != "ATOM":
                continue
            if int(line[22:26].strip()) != current_pos:
                current_pos = int(line[22:26].strip())
                score_line = score_lines.pop(0)
            new_line = line[0:60] + score_line + "           " + line.strip()[-1] + "  \n"
            new_pdb += new_line
        new_pdb += "TER\n"

        with open("{}/pred/{}.pdb".format(outpath, ID), "w") as f:
            f.write(new_pdb)
        '''


    with open(outpath + "esmfold_pred.log", "r") as f:
        lines = f.readlines()

    info_dict = {}
    for line in lines:
        if "pLDDT" in line:
            ID_len, pLDDT, pTM = line.strip().split("|")[-1].strip().split(",")[0:3]
            ID = ID_len.strip().split()[3]
            length = ID_len.strip().split()[6]
            pLDDT = float(pLDDT.strip().split()[1])
            pTM = float(pTM.strip().split()[1])
            info_dict[ID] = [length, pLDDT, pTM]

    # bellow is used to save the prediction into one overview.txt file.
    entry_info = "ID\tLength\tpLDDT\tpTM\tDNA_Binding\tRNA_Binding\tPeptide_Binding\tProtein_Binding\tATP_Binding\tHEM_Binding\tZN_Binding\tCA_Binding\tMG_Binding\tMN_Binding\n"
    for ID in predictions:
        Length, pLDDT, pTM = info_dict[ID]
        binding_scores = GPSite_binding_scores[ID]
        binding_scores = np.array(binding_scores)[index] # switch order to DNA, RNA, PEP ...
        entry = "{}\t{}\t{}\t{:.3f}".format(ID, Length, pLDDT, pTM)
        for score in binding_scores:
            entry += "\t{:.3f}".format(score)
        entry_info += (entry + "\n")

    with open("{}/pred/overview.txt".format(outpath), "w") as f:
        f.write(entry_info)

    os.system("rm {}/esmfold_pred.log".format(outpath))

if __name__ == '__main__':
    x = pt.randn(10, 3)
    ids_topk = pt.randint(0, 10, (10, 4)).long()
    ind, attr = unpack_edge_features(x, ids_topk)
    print(f'the shape of ind0: {ind[0].shape} ind1: {ind[1].shape} and the edge attr: {attr.shape}')