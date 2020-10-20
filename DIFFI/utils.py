import numpy as np
from sklearn.ensemble import IsolationForest

from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils.validation import _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import numpy as np
import math


def decision_function_single_tree(iforest, tree_idx, X):
    return _score_samples(iforest, tree_idx, X) - iforest.offset_


def _score_samples(iforest, tree_idx, X):
    if iforest.n_features_ != X.shape[1]:
        raise ValueError("Number of features of the model must "
                         "match the input. Model n_features is {0} and "
                         "input n_features is {1}."
                         "".format(iforest.n_features_, X.shape[1]))
    return -_compute_chunked_score_samples(iforest, tree_idx, X)


def _compute_chunked_score_samples(iforest, tree_idx, X):
    n_samples = _num_samples(X)
    if iforest._max_features == X.shape[1]:
        subsample_features = False
    else:
        subsample_features = True
    chunk_n_rows = get_chunk_n_rows(row_bytes=16 * iforest._max_features,
                                    max_n_rows=n_samples)
    slices = gen_batches(n_samples, chunk_n_rows)
    scores = np.zeros(n_samples, order="f")
    for sl in slices:
        scores[sl] = _compute_score_samples_single_tree(
            iforest, tree_idx, X[sl], subsample_features)
    return scores


def _compute_score_samples_single_tree(iforest, tree_idx, X,
                                       subsample_features):
    n_samples = X.shape[0]
    depths = np.zeros(n_samples, order="f")
    tree = iforest.estimators_[tree_idx]
    features = iforest.estimators_features_[tree_idx]
    X_subset = X[:, features] if subsample_features else X
    leaves_index = tree.apply(X_subset)
    node_indicator = tree.decision_path(X_subset)
    n_samples_leaf = tree.tree_.n_node_samples[leaves_index]
    depths += (np.ravel(node_indicator.sum(axis=1)) +
               _average_path_length(n_samples_leaf) - 1.0)
    scores = 2**(-depths / (1 * _average_path_length([iforest.max_samples_])))
    return scores


def _get_iic(estimator: IsolationForest,
             predictions,
             is_leaves,
             adjust_iic,
             epsilon=0,
             desired_min=0.5,
             desired_max=1.0):
    """
    Comptue the Induced balance coefficients

    Arguments
    ---------
    estimator:
    """
    n_nodes = estimator.tree_.node_count
    lambda_ = np.zeros(n_nodes)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    # compute samples in each node
    node_indicator_all_samples = estimator.decision_path(predictions).toarray()
    num_samples_in_node = np.sum(node_indicator_all_samples, axis=0)
    # ASSIGN INDUCED IMBALANCE COEFFICIENTS (IIC)
    for node in range(n_nodes):
        # compute relevant quantities for current node
        num_samples_in_current_node = num_samples_in_node[node]
        num_samples_in_left_children = num_samples_in_node[children_left[node]]
        num_samples_in_right_children = num_samples_in_node[
            children_right[node]]
        # if there is only 1 feasible split or node is leaf -> no IIC is assigned
        if num_samples_in_current_node == 0 or num_samples_in_current_node == 1 or is_leaves[
                node]:
            lambda_[node] = -1
        # if useless split -> assign epsilon
        elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
            lambda_[node] = epsilon
        else:
            if num_samples_in_current_node % 2 == 0:  # even
                current_min = 0.5
            else:  # odd
                current_min = math.ceil(num_samples_in_current_node /
                                        2) / num_samples_in_current_node
            current_max = (num_samples_in_current_node -
                           1) / num_samples_in_current_node
            tmp = np.max([
                num_samples_in_left_children, num_samples_in_right_children
            ]) / num_samples_in_current_node
            if adjust_iic and current_min != current_max:
                lambda_[node] = ((tmp - current_min) /
                                 (current_max - current_min)) * (
                                     desired_max - desired_min) + desired_min
            else:
                lambda_[node] = tmp
    return lambda_


def update_counters(estimator: IsolationForest, X_ib, node_depth: list,
                    is_leaves: bool, adjust_iic: bool, cfi_ib: np.array,
                    counter_ib: np.array):

    feature = estimator.tree_.feature
    lambda_ib = _get_iic(estimator, X_ib, is_leaves, adjust_iic)
    node_indicator_ib = estimator.decision_path(X_ib).toarray()

    for i in range(len(X_ib)):
        path = np.where(node_indicator_ib[i] == 1)[0].tolist()
        depth = node_depth[path[-1]]
        for node in path:
            current_feature = feature[node]
            if lambda_ib[node] == -1:
                continue
            cfi_ib[current_feature] += (1 / depth) * lambda_ib[node]
            counter_ib[current_feature] += 1


def compute_node_depth(tree):
    
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    # compute node depths
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # if we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    return node_depth, is_leaves


def diffi_inbag(iforest: IsolationForest, X, adjust_iic=True):
    # initialization
    num_feat = X.shape[1]
    cfi_outliers_ib = np.zeros(num_feat, dtype=np.float)
    cfi_inliers_ib = np.zeros(num_feat, dtype=np.float)
    counter_outliers_ib = np.zeros(num_feat, dtype=np.int)
    counter_inliers_ib = np.zeros(num_feat, dtype=np.int)

    # for every iTree in the iForest
    for k, estimator in enumerate(iforest.estimators_):
        # get in-bag samples indices
        in_bag_sample = list(iforest.estimators_samples_[k])
        # get in-bag samples (predicted inliers and predicted outliers)
        X_ib = X[in_bag_sample, :]
        as_ib = decision_function_single_tree(iforest, k, X_ib)
        X_outliers_ib = X_ib[np.where(as_ib < 0)]
        X_inliers_ib = X_ib[np.where(as_ib > 0)]
        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
            continue
        # compute relevant quantities
        node_depth, is_leaves = compute_node_depth(estimator.tree_)

        update_counters(iforest, X_outliers_ib, node_depth, is_leaves,
                        adjust_iic, cfi_outliers_ib, counter_outliers_ib)

        update_counters(iforest, X_inliers_ib, node_depth, is_leaves,
                        adjust_iic, cfi_inliers_ib, counter_inliers_ib)

    # compute FI
    fi_outliers_ib = np.where(counter_outliers_ib > 0,
                              cfi_outliers_ib / counter_outliers_ib, 0)
    fi_inliers_ib = np.where(counter_inliers_ib > 0,
                             cfi_inliers_ib / counter_inliers_ib, 0)
    fi_ib = fi_outliers_ib / fi_inliers_ib
    return fi_ib


def logarithmic_scores(fi):
    """
    Parameters
    ----------
    fi: np.array
        fi is a (N x p) matrix, where N is the number of runs and p is the number of features
    """
    
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





def local_diffi_one_sample(iforest:IsolationForest, x):
    """
    Compute fi for one sample

    Parameters
    ----------

    """
    # initialization 
    cfi = np.zeros(len(x), dtype='float')
    counter = np.zeros(len(x), dtype='int')
    max_depth = int(np.ceil(np.log2(iforest.max_samples)))
    for estimator in iforest.estimators_:
        feature = estimator.tree_.feature
        node_depth, is_leaves= compute_node_depth(estimator.tree_)
        
        x = x.reshape(1,-1)
        node_indicator = estimator.decision_path(x)
        node_indicator_array = node_indicator.toarray()
        path = list(np.where(node_indicator_array == 1)[1])
        leaf_depth = node_depth[path[-1]]
        for node in path:
            if is_leaves[node]:
                continue
            current_feature = feature[node] 
            cfi[current_feature] += (1 / leaf_depth) - (1 / max_depth)
            counter[current_feature] += 1
    # compute FI    
    fi = [cfi[i] / counter[i] if counter[i] != 0 else 0 
          for i in range(len(cfi)) ]
    return np.array(fi)
    

def local_diffi(iforest:IsolationForest, X):
    fi = []
    ord_idx = []
    for i in range(X.shape[0]):
        x_curr = X[i, :]
        fi_curr = local_diffi_one_sample(iforest, x_curr)
        fi.append(fi_curr)
        ord_idx_curr = np.argsort(fi_curr)[::-1]
        ord_idx.append(ord_idx_curr)
    fi = np.vstack(fi)
    ord_idx = np.vstack(ord_idx)
    return fi, ord_idx