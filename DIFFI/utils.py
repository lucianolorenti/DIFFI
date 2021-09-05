import numpy as np
from sklearn.ensemble import IsolationForest

from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils.validation import _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.tree._classes import BaseDecisionTree


def decision_function_single_tree(iforest, tree_idx, X):
    return _score_samples(iforest, tree_idx, X) - iforest.offset_


def _score_samples(iforest, tree_idx, X):
    if iforest.n_features_ != X.shape[1]:
        raise ValueError(
            "Number of features of the model must "
            "match the input. Model n_features is {0} and "
            "input n_features is {1}."
            "".format(iforest.n_features_, X.shape[1])
        )
    return -_compute_chunked_score_samples(iforest, tree_idx, X)


def _compute_chunked_score_samples(iforest, tree_idx, X):
    n_samples = _num_samples(X)
    if iforest._max_features == X.shape[1]:
        subsample_features = False
    else:
        subsample_features = True
    chunk_n_rows = get_chunk_n_rows(
        row_bytes=16 * iforest._max_features, max_n_rows=n_samples
    )
    slices = gen_batches(n_samples, chunk_n_rows)
    scores = np.zeros(n_samples, order="f")
    for sl in slices:
        scores[sl] = _compute_score_samples_single_tree(
            iforest, tree_idx, X[sl], subsample_features
        )
    return scores


def _compute_score_samples_single_tree(iforest, tree_idx, X, subsample_features):
    n_samples = X.shape[0]
    depths = np.zeros(n_samples, order="f")
    tree = iforest.estimators_[tree_idx]
    features = iforest.estimators_features_[tree_idx]
    X_subset = X[:, features] if subsample_features else X
    leaves_index = tree.apply(X_subset)
    node_indicator = tree.decision_path(X_subset)
    n_samples_leaf = tree.tree_.n_node_samples[leaves_index]
    depths += (
        np.ravel(node_indicator.sum(axis=1))
        + _average_path_length(n_samples_leaf)
        - 1.0
    )
    scores = 2 ** (-depths / (1 * _average_path_length([iforest.max_samples_])))
    return scores


def logarithmic_scores(fi):
    """
    Parameters
    ----------
    fi: np.array
        fi is a (N x p) matrix, where N is the number of runs and p is the number of features
    """

    num_feats = fi.shape[1]
    p = np.arange(1, num_feats + 1, 1)
    log_s = [1 - (np.log(x) / np.log(num_feats)) for x in p]
    scores = np.zeros(num_feats)
    for i in range(fi.shape[0]):
        sorted_idx = np.flip(np.argsort(fi[i, :]))
        for j in range(num_feats):
            curr_feat = sorted_idx[j]
            if fi[i, curr_feat] > 0:
                scores[curr_feat] += log_s[j]
    return scores


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
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    return node_depth, is_leaves


def local_diffi_one_sample(iforest: IsolationForest, x):
    """
    Compute fi for one sample

    Parameters
    ----------

    """
    # initialization
    cfi = np.zeros(len(x), dtype="float")
    counter = np.zeros(len(x), dtype="int")
    max_depth = int(np.ceil(np.log2(iforest.max_samples)))
    for estimator in iforest.estimators_:
        feature = estimator.tree_.feature
        node_depth, is_leaves = compute_node_depth(estimator.tree_)

        x = x.reshape(1, -1)
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
    fi = [cfi[i] / counter[i] if counter[i] != 0 else 0 for i in range(len(cfi))]
    return np.array(fi)
