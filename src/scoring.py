import numpy as np
import torch as pt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score
from torcheval.metrics.functional import binary_auprc, binary_f1_score, binary_precision, binary_recall

epsilon = 1e-10

def binary_classification(y, q):
    # total positive and negatives
    TP = pt.sum(q * y, dim=0)
    TN = pt.sum((1.0-q) * (1.0-y), dim=0)
    FP = pt.sum(q * (1.0-y), dim=0)
    FN = pt.sum((1.0-q) * y, dim=0)
    P = pt.sum(y, dim=0)
    N = pt.sum(1.0-y, dim=0)

    return TP, TN, FP, FN, P, N



def acc(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN + epsilon)



def ppv(TP, FP, P):
    v = TP / (TP + FP + epsilon) 

    v[~(P>0)] = np.nan  # no positives -> undefined
    return v



def npv(TN, FN, N):
    v = TN / (TN + FN + epsilon)
    v[~(N>0)] = np.nan  # no negatives -> undefined
    return v



def tpr(TP, FN):
    v = TP / (TP + FN + epsilon)
    v[pt.isinf(v)] = np.nan
    return v



def fpr(FP, TN):
    v = FP / (FP + TN + epsilon)
    v[pt.isinf(v)] = np.nan
    return v


def tnr(TN, FP):
    v = TN / (TN + FP + epsilon)
    v[pt.isinf(v)] = np.nan
    return v


def mcc(TP, TN, FP, FN):
    v = ((TP*TN) - (FP*FN)) / (pt.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + epsilon)
    v[pt.isinf(v)] = np.nan
    return v

def f1(y, p, P, N):
    # p = pt.round(p)
    m = (P > 0) & (N > 0)
    v = pt.zeros(y.shape[1], dtype=pt.float32, device=y.device)
    if pt.any(m):
        idx = pt.arange(m.shape[0])[m]
        for i in idx:
            v[i] = binary_f1_score(p[:,i], y[:,i])
    v[~m] = np.nan
    return v


def pr_auc(y, p, P, N):
    m = (P > 0) & (N > 0)
    v = pt.zeros(y.shape[1], dtype=pt.float32, device=y.device)
    if pt.any(m):
        a = np.array(average_precision_score(y[:,m].cpu().numpy(), p[:,m].cpu().numpy(), average=None))
        v[m] = pt.from_numpy(a).float().to(y.device)
    v[~m] = np.nan
    return v

def roc_auc(y, p, P, N):
    m = (P > 0) & (N > 0)
    v = pt.zeros(y.shape[1], dtype=pt.float32, device=y.device)
    if pt.any(m):
        a = np.array(roc_auc_score(y[:,m].cpu().numpy(), p[:,m].cpu().numpy(), average=None))
        v[m] = pt.from_numpy(a).float().to(y.device)
    v[~m] = np.nan
    return v



def nanmean(x):
    return pt.nansum(x, dim=0) / pt.sum(~pt.isnan(x), dim=0)


bc_score_names = [
    'Acc','PPV(Precision)','NPV','TPR(Recall)',
    'TNR','MCC','ROC AUC','STD','PR AUC','F1','FPR'
]

def bc_scoring(y, p):
    """
    Compute binary classification scores

    Args:
        y (torch.Tensor): true labels, [num_residues, num_binding_types]
        p (torch.Tensor): predicted labels, [num_residues, num_binding_types]
        
    Returns
    -------
        scores (torch.Tensor): scores, [num_metrics, num_binding_types]
    """
    # prediction
    q = pt.round(p)

    # binary classification
    TP, TN, FP, FN, P, N = binary_classification(y, q)

    # compute and pack scores
    scores = pt.stack([
        acc(TP, TN, FP, FN),
        ppv(TP, FP, P),
        npv(TN, FN, N),
        tpr(TP, FN),
        tnr(TN, FP),
        mcc(TP, TN, FP, FN),
        roc_auc(y, p, P, N),
        pt.std(p, dim=0),
        pr_auc(y, p, P, N),#auprc(y, p), 
        f1(y, p, P, N),
        fpr(FP, TN)
    ])

    return scores

def reg_scoring(y, p):
    return {
        'mse': float(pt.mean(pt.square(y - p)).cpu().numpy()),
        'mae': float(pt.mean(pt.abs(y - p)).cpu().numpy()),
        'rmse': float(pt.sqrt(pt.mean(pt.square(y - p))).cpu().numpy()),
        'pcc': pearsonr(y.cpu().numpy(), p.cpu().numpy())[0] if not pt.allclose(y,y[0]) else np.nan,
        'std': float(pt.std(p).cpu().numpy()),
    }






