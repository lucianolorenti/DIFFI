import numpy as np 
import pandas as pd 
import os 
import time
from scipy.io import loadmat 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import shap
import interpretability_module as interp


def local_diffi_batch(iforest, X):
    fi = []
    ord_idx = []
    exec_time = []
    for i in range(X.shape[0]):
        x_curr = X[i, :]
        fi_curr, exec_time_curr = interp.local_diffi(iforest, x_curr)
        fi.append(fi_curr)
        ord_idx_curr = np.argsort(fi_curr)[::-1]
        ord_idx.append(ord_idx_curr)
        exec_time.append(exec_time_curr)
    fi = np.vstack(fi)
    ord_idx = np.vstack(ord_idx)
    return fi, ord_idx, exec_time


def local_shap_batch(iforest, X):
    fi = []
    ord_idx = []
    exec_time = []
    for i in range(X.shape[0]):
        x_curr = X[i, :]
        start = time.time()
        explainer = shap.TreeExplainer(iforest)
        shap_values = explainer.shap_values(x_curr)
        fi_curr = np.abs(shap_values)
        exec_time_curr = time.time() - start
        fi.append(fi_curr)
        ord_idx_curr = np.argsort(fi_curr)[::-1]
        ord_idx.append(ord_idx_curr)
        exec_time.append(exec_time_curr)
    fi = np.vstack(fi)
    ord_idx = np.vstack(ord_idx)
    return fi, ord_idx, exec_time


def logarithmic_scores(fi):
    # fi is a (N x p) matrix, where N is the number of runs and p is the number of features
    num_feats = fi.shape[1]
    p = np.arange(1, num_feats + 1, 1)
    log_s = [1 - (np.log(x)/np.log(num_feats)) for x in p]
    scores = np.zeros(num_feats)
    for i in range(fi.shape[0]):
        sorted_idx = np.flip(np.argsort(fi[i,:]))
        for j in range(num_feats):
            curr_feat = sorted_idx[j]
            if fi[i,curr_feat]>0:
                scores[curr_feat] += log_s[j]
    return scores 
  

  

def get_fs_dataset(dataset_id, seed):
    if dataset_id == 'cardio' or dataset_id == 'ionosphere' or dataset_id == 'letter' \
        or dataset_id == 'lympho' or dataset_id == 'musk' or dataset_id == 'satellite':
        mat = loadmat(os.path.join(os.getcwd(), 'data', 'ufs', dataset_id + '.mat'))
        X = mat['X']
        y = mat['y'].squeeze()
        print('\nLoaded {} dataset: {} samples, {} features.'.format(dataset_id, X.shape[0], X.shape[1]))
    y = y.astype('int')
    contamination = len(y[y == 1])/len(y)
    print('{:2.2f} percent outliers.'.format(contamination*100))
    X, y = shuffle(X, y, random_state=seed)
    return X, y, contamination  


def diffi_ranks(X, y, n_trees, max_samples, n_iter):
    f1_all, fi_diffi_all = [], []
    for k in range(n_iter):
        # ISOLATION FOREST
        # fit the model
        iforest = IsolationForest(n_estimators= n_trees, max_samples=max_samples, 
                                  contamination='auto', random_state=k)
        iforest.fit(X)
        # get predictions
        y_pred = np.array(iforest.decision_function(X) < 0).astype('int')
        # get performance metrics
        f1_all.append(f1_score(y, y_pred))
        # diffi
        fi_diffi, _ = interp.diffi_ib(iforest, X, adjust_iic=True)
        fi_diffi_all.append(fi_diffi)
    # compute avg F1 
    avg_f1 = np.mean(f1_all)
    # compute the scores
    fi_diffi_all = np.vstack(fi_diffi_all)
    scores = logarithmic_scores(fi_diffi_all)
    sorted_idx = np.flip(np.argsort(scores))
    return sorted_idx, avg_f1


def fs_datasets_hyperparams(dataset):
    data = {
            # cardio
            ('cardio'): {'contamination': 0.1, 'max_samples': 64, 'n_estimators': 150},
            # ionosphere
            ('ionosphere'): {'contamination': 0.2, 'max_samples': 256, 'n_estimators': 100},
            # lympho
            ('lympho'): {'contamination': 0.05, 'max_samples': 64, 'n_estimators': 150},
            # letter
            ('letter'):  {'contamination': 0.1, 'max_samples': 256, 'n_estimators': 50},
            # musk
            ('musk'): {'contamination': 0.05, 'max_samples': 128, 'n_estimators': 100},
            # satellite
            ('satellite'): {'contamination': 0.15, 'max_samples': 64, 'n_estimators': 150}
            }
    return data[dataset]