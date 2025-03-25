# import sys
from datetime import datetime
from src.data_encoding import categ_to_resnames


config_data = {
    'dataset_filepath': "/home/yw7bh/data/Projects/FunBench/ProBST/experiments/contacts_rr5A_64nn_8192.h5",
    'train_selection_filepath': "/home/yw7bh/data/Projects/FunBench/ProBST/datasets/subunits_train_set.txt",
    'test_selection_filepath': "/home/yw7bh/data/Projects/FunBench/ProBST/datasets/subunits_validation_set.txt",
    'new_selection_filepath': "/home/yw7bh/data/Projects/FunBench/construct_testset/data/nonredundant_thre-2022-01-01_seq-id-0.3.txt",
    'max_ba': 1,
    'max_size': 1024*8,   # original is 8
    'min_num_res': 48,
    'l_types': categ_to_resnames['protein'],
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna']+categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ],
    # 'r_types': [[c] for c in categ_to_resnames['protein']],
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'model_'+tag,
    'output_dir': 'save',
    'reload': True,
    'device': "cpu",   # 'cuda:0',
    'num_epochs': 500,
    'batch_size': 1,
    'log_step': 1225,   # orginal 1024, 897 for 114658 training samples with 1792 batches, while 1214 for training samples with 2427 batches
    'eval_step': 1225*1,    # orginal 1024*1, 897 for 114658 training samples with 1792 batches, while 1214 for training samples with 2427 batches
    'eval_size': 636*1,    # orginal 1024*1
    'learning_rate': 1e-3,
    'pos_weight_factor': 0.5,
    'comment': "",
}
