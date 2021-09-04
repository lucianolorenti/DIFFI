from time import time

import numpy as np
import shap
from sklearn.ensemble import IsolationForest


def local_shap_batch(iforest: IsolationForest, X):
    fi = []
    ord_idx = []

    for i in range(X.shape[0]):
        x_curr = X[i, :]
        explainer = shap.TreeExplainer(iforest)
        shap_values = explainer.shap_values(x_curr)
        fi_curr = np.abs(shap_values)
        fi.append(fi_curr)
        ord_idx_curr = np.argsort(fi_curr)[::-1]
        ord_idx.append(ord_idx_curr)
    fi = np.vstack(fi)
    ord_idx = np.vstack(ord_idx)
    return fi, ord_idx
