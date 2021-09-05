from typing import Optional, Tuple
from DIFFI.utils import compute_node_depth, decision_function_single_tree, logarithmic_scores
import numpy as np
from sklearn.ensemble import IsolationForest

from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils.validation import _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.tree._classes import BaseDecisionTree


def _get_iic(
    estimator: IsolationForest,
    predictions,
    is_leaves,
    adjust_iic,
    epsilon=0,
    desired_min=0.5,
    desired_max=1.0,
):
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

    for node in range(n_nodes):
        # compute relevant quantities for current node
        num_samples_in_current_node = num_samples_in_node[node]
        num_samples_in_left_children = num_samples_in_node[children_left[node]]
        num_samples_in_right_children = num_samples_in_node[children_right[node]]
        # if there is only 1 feasible split or node is leaf -> no IIC is assigned
        if (
            num_samples_in_current_node == 0
            or num_samples_in_current_node == 1
            or is_leaves[node]
        ):
            lambda_[node] = -1
        # if useless split -> assign epsilon
        elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
            lambda_[node] = epsilon
        else:
            if num_samples_in_current_node % 2 == 0:  # even
                current_min = 0.5
            else:  # odd
                current_min = (
                    math.ceil(num_samples_in_current_node / 2)
                    / num_samples_in_current_node
                )
            current_max = (
                num_samples_in_current_node - 1
            ) / num_samples_in_current_node
            tmp = (
                np.max([num_samples_in_left_children, num_samples_in_right_children])
                / num_samples_in_current_node
            )
            if adjust_iic and current_min != current_max:
                lambda_[node] = ((tmp - current_min) / (current_max - current_min)) * (
                    desired_max - desired_min
                ) + desired_min
            else:
                lambda_[node] = tmp
    return lambda_


def update_counters(
    estimator: BaseDecisionTree,
    X_ib,
    node_depth: list,
    is_leaves: bool,
    adjust_iic: bool,
    cfi_ib: np.array,
    counter_ib: np.array,
):

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

        update_counters(
            estimator,
            X_outliers_ib,
            node_depth,
            is_leaves,
            adjust_iic,
            cfi_outliers_ib,
            counter_outliers_ib,
        )

        update_counters(
            estimator,
            X_inliers_ib,
            node_depth,
            is_leaves,
            adjust_iic,
            cfi_inliers_ib,
            counter_inliers_ib,
        )

    # compute FI
    fi_outliers_ib = np.where(
        counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0
    )
    fi_inliers_ib = np.where(
        counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0
    )
    fi_ib = fi_outliers_ib / fi_inliers_ib
    return fi_ib


def global_diffi_ranks(
    X: np.ndarray, y: Optional[np.ndarray] = None, n_iter: int = 5, **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute the global DIFFI score

    Parameters
    ----------
    X : np.ndarray
        The input data
    y : np.ndarray
        The target
    n_iter : int
        maximun number of iterations

    Returns
    -------
    [type]
        [description]
    """
    f1_all, fi_diffi_all = [], []
    for k in range(n_iter):

        iforest = IsolationForest(
            **kwargs,
            contamination="auto",
        )
        iforest.fit(X)

        y_pred = np.array(iforest.decision_function(X) < 0).astype("int")

        if y is not None:
            f1_all.append(f1_score(y, y_pred))

        fi_diffi = diffi_inbag(iforest, X, adjust_iic=True)
        fi_diffi_all.append(fi_diffi)
    if y is not None:
        avg_f1 = np.mean(f1_all)
    else:
        avg_f1 = None

    fi_diffi_all = np.vstack(fi_diffi_all)
    scores = logarithmic_scores(fi_diffi_all)
    sorted_idx = np.flip(np.argsort(scores))

    return sorted_idx, avg_f1
